/* verilator lint_off DECLFILENAME */
//------------------------------------------------------------------------------
// Ada SM Tensor Core Educational Model  (RTL + Testbench)
//------------------------------------------------------------------------------
// * Fixed systolic‑array bugs (reset edge & psum routing)
// * Added optional VCD/FST tracing via $dumpfile/$dumpvars or +trace
// * Cleaned up Verilator warnings (blocking assignments in initial blocks)
//------------------------------------------------------------------------------
`timescale 1ns/1ps

//---------------------------------------------
// Utility: ceil‑log2 (simple integer function)
//---------------------------------------------
function automatic integer clog2(input integer value);
    integer i; begin
        clog2 = 0;
        for (i = value-1; i > 0; i = i >> 1) clog2++;
    end
endfunction

//==========================================================
//  Processing Element (PE) – scalar MAC
//==========================================================
module pe #(parameter WIDTH = 16) (
    input  wire                 clk,
    input  wire                 rst,
    input  wire [WIDTH-1:0]     a_in,
    input  wire [WIDTH-1:0]     b_in,
    input  wire [2*WIDTH-1:0]   psum_in,
    output reg  [WIDTH-1:0]     a_out,
    output reg  [WIDTH-1:0]     b_out,
    output reg  [2*WIDTH-1:0]   psum_out
);
    always @(posedge clk) begin
        if (rst) begin
            a_out    <= '0;
            b_out    <= '0;
            psum_out <= '0;
        end else begin
            a_out    <= a_in;
            b_out    <= b_in;
            psum_out <= psum_in + a_in * b_in;
        end
    end
endmodule

//==========================================================
//  tensor_core – 8×8 systolic array, k‑depth=4
//==========================================================
module tensor_core #(
    parameter WIDTH   = 16,
    parameter SIZE    = 8,
    parameter KDEPTH  = 4
)(
    input  wire                         clk,
    input  wire                         rst,
    input  wire [SIZE*WIDTH-1:0]        a_left,
    input  wire [SIZE*WIDTH-1:0]        b_top,
    input  wire                         a_valid,
    input  wire                         b_valid,
    output wire [SIZE*2*WIDTH-1:0]      c_bottom,
    output wire                         c_valid
);
    // Shift registers holding K‑slice history
    reg [SIZE*WIDTH-1:0] a_shift[KDEPTH-1:0];
    reg [SIZE*WIDTH-1:0] b_shift[KDEPTH-1:0];
    reg [clog2(KDEPTH)-1:0] k_ctr;

    always @(posedge clk) begin
        if (rst) begin
            a_shift <= '{default:'0};
            b_shift <= '{default:'0};
            k_ctr   <= '0;
        end else if (a_valid & b_valid) begin
            a_shift <= {a_shift[KDEPTH-2:0], a_left};
            b_shift <= {b_shift[KDEPTH-2:0], b_top};
            k_ctr   <= (k_ctr == KDEPTH-1) ? 0 : k_ctr + 1;
        end
    end

    // Operand & psum buses (extra column for right‑shift)
    wire [WIDTH-1:0]     a_bus [0:SIZE] [0:SIZE-1];
    wire [WIDTH-1:0]     b_bus [0:SIZE-1] [0:SIZE];
    wire [2*WIDTH-1:0]   psum_bus [0:SIZE-1] [0:SIZE];

    genvar i,j;
    generate
        // feed left/top boundaries from current k‑slice
        for (i=0;i<SIZE;i++) begin : FEED
            assign a_bus[0][i] = a_shift[k_ctr][i*WIDTH +: WIDTH];
            assign b_bus[i][0] = b_shift[k_ctr][i*WIDTH +: WIDTH];
        end
        // grid of PEs
        for (i=0;i<SIZE;i++) begin : ROW
            for (j=0;j<SIZE;j++) begin : COL
                pe #(.WIDTH(WIDTH)) u_pe (
                    .clk      (clk),
                    // reset on FIRST slice (k_ctr==0) so each tile clears once
                    .rst      (rst | (a_valid & b_valid & (k_ctr==0))),
                    .a_in     (a_bus[j][i]),
                    .b_in     (b_bus[i][j]),
                    .psum_in  ((j==0)? '0 : psum_bus[i][j]),
                    .a_out    (a_bus[j+1][i]),
                    .b_out    (b_bus[i][j+1]),
                    .psum_out (psum_bus[i][j+1])
                );
            end
        end
    endgenerate

    // collect final column (SIZE) to bottom edge
    generate
        for (i=0;i<SIZE;i++) begin : COLLECT
            assign c_bottom[i*2*WIDTH +: 2*WIDTH] = psum_bus[i][SIZE];
        end
    endgenerate

    // latency: SIZE + KDEPTH cycles from slice‑0 arrival
    localparam LAT = SIZE + KDEPTH;
    reg [clog2(LAT)-1:0] lat_ctr;
    reg                  out_phase;
    always @(posedge clk) begin
        if (rst) begin
            lat_ctr <= 0; out_phase <= 0;
        end else if (a_valid & b_valid & (k_ctr==KDEPTH-1)) begin
            out_phase <= 1; lat_ctr <= 0;
        end else if (out_phase) begin
            lat_ctr <= lat_ctr + 1;
            if (lat_ctr == LAT-1) out_phase <= 0;
        end
    end
    assign c_valid = out_phase & (lat_ctr == LAT-1);
endmodule

//==========================================================
//  ada_sm_tensor – 4 Tensor Cores per SM
//==========================================================
module ada_sm_tensor #(
    parameter WIDTH = 16,
    parameter SIZE  = 8,
    parameter CORES = 4
)(
    input  wire                                   clk,
    input  wire                                   rst,
    input  wire [CORES*SIZE*WIDTH-1:0]            a_left_all,
    input  wire [CORES*SIZE*WIDTH-1:0]            b_top_all,
    input  wire [CORES-1:0]                       in_valid,
    output wire [CORES*SIZE*2*WIDTH-1:0]          c_bottom_all,
    output wire [CORES-1:0]                       c_valid_all
);
    genvar core;
    generate
        for (core=0; core<CORES; core++) begin : CORE
            tensor_core #(.WIDTH(WIDTH), .SIZE(SIZE)) u_tc (
                .clk      (clk),
                .rst      (rst),
                .a_left   (a_left_all[core*SIZE*WIDTH +: SIZE*WIDTH]),
                .b_top    (b_top_all [core*SIZE*WIDTH +: SIZE*WIDTH]),
                .a_valid  (in_valid[core]),
                .b_valid  (in_valid[core]),
                .c_bottom (c_bottom_all[core*SIZE*2*WIDTH +: SIZE*2*WIDTH]),
                .c_valid  (c_valid_all[core])
            );
        end
    endgenerate
endmodule

//**********************************************************
//  TESTBENCH – 32‑lane warp drives the SM, optional VCD
//**********************************************************
module tb_ada_sm_tensor;
    localparam WIDTH=16, SIZE=8, KDEPTH=4, CORES=4;

    // clock gen (200 MHz)
    logic clk=0; always #2.5 clk=~clk;
    logic rst=1;

    logic [CORES*SIZE*WIDTH-1:0] a_left_flat='0;
    logic [CORES*SIZE*WIDTH-1:0] b_top_flat='0;
    logic [CORES-1:0]            in_valid='0;
    logic [CORES*SIZE*2*WIDTH-1:0] c_bottom_all;
    logic [CORES-1:0]              c_valid_all;

    ada_sm_tensor #(.WIDTH(WIDTH),.SIZE(SIZE)) dut(
        .clk(clk),.rst(rst),
        .a_left_all(a_left_flat),
        .b_top_all (b_top_flat),
        .in_valid  (in_valid),
        .c_bottom_all(c_bottom_all),
        .c_valid_all(c_valid_all));

    // ---------------- VCD / FST dump ----------------
    initial begin
        if ($test$plusargs("DUMP")) begin
            $dumpfile("wave.vcd");
            $dumpvars(0, tb_ada_sm_tensor);
        end
    end

    // ---------------- helper function ---------------
    function automatic [SIZE*WIDTH-1:0] make_slice(input int k_idx, bit transpose);
        logic [SIZE*WIDTH-1:0] slice;
        for (int i=0;i<SIZE;i++)
            slice[i*WIDTH +: WIDTH] = ((transpose? i : k_idx)==(transpose? k_idx : i)) ? 16'd1 : 16'd0;
        return slice;
    endfunction

    // ---------------- stimulus ----------------------
    initial begin
        repeat (4) @(posedge clk);
        rst = 0;

        // stream 4 k‑slices
        for (int k=0; k<KDEPTH; k++) begin
            @(posedge clk);
            in_valid = '1;                                   // all 4 cores
            for (int c=0;c<CORES;c++) begin
                int off = c*SIZE*WIDTH;
                a_left_flat[off +: SIZE*WIDTH] = make_slice(k,0);
                b_top_flat [off +: SIZE*WIDTH] = make_slice(k,1);
            end
        end
        @(posedge clk) in_valid = '0;

        // wait SIZE+KDEPTH cycles (12)
        repeat (SIZE+KDEPTH) @(posedge clk);

        $display("c_valid_all = %b", c_valid_all);
        for (int row=0; row<SIZE; row++) begin
            $write("Row%0d: ", row);
            for (int col=0; col<SIZE; col++) begin
                int base=(row*SIZE+col)*(2*WIDTH);
                $write("%0d ", c_bottom_all[base +: 2*WIDTH]);
            end
            $write("\n");
        end
        $finish;
    end
endmodule
