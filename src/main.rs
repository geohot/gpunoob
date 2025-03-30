use std::{cmp::max, cmp::min, error::Error};
use plotters::prelude::*;
use ocl::{ProQue, Buffer, MemFlags};

// dims = G*L (g*num_threads + l)
//            OpenCL        CUDA       HIP                  Metal
// G cores   (get_group_id, blockIdx,  __ockl_get_group_id, threadgroup_position_in_grid)
// L threads (get_local_id, threadIdx, __ockl_get_local_id, thread_position_in_threadgroup)

// 128 cores, 1 thread each
// 64 cores, 2 threads each

// GPUs have warps. Warps are groups of threads, and all modern GPUs have them as 32 threads.
// GPUs are multicore processors with 32 threads

// On NVIDIA, cores are streaming multiprocessors.
//   AD102 (4090) has 144 SMs with 128 threads each
// On AMD, cores are compute units.
//   7900XTX has 96 CUs with 64 threads each
// On Apple, ???
//   M3 Max has a 40 core GPU, 640 "Execution Units", 5120 "ALUs"
//   Measured. 640 EUs with 32 threads each (20480 threads total)

// SIMD - Single Instruction Multiple Data
//    vector registers
//    float<32> (1024 bits)
//    c = a + b (on vector registers, this is a single add instruction on 32 pieces of data)

// SIMT - Single Instruction Multiple Thread
//    similar to SIMD, but load/stores are different
//    you only declare "float", but behind the scenes it's float<32>
//    load stores are implicit scatter gather, whereas on SIMD it's explicit

fn main() -> Result<(), Box<dyn Error>> {
    // Kernel source code (OpenCL C)
            //__global const float* a,
            //__global const float* b,

    // Input data
    //let a_data = vec![1.0f32; 128];
    //let b_data = vec![2.0f32; 128];

    // Create buffers
    //let a_buffer = proque.create_buffer::<f32>()?;
    //let b_buffer = proque.create_buffer::<f32>()?;
    // Write data to device buffers
    //a_buffer.cmd().write(&a_data).enq()?;
    //b_buffer.cmd().write(&b_data).enq()?;
        //.arg(&a_buffer)
        //.arg(&b_buffer)

    // this kernel takes 15ms to execute
    // 1e6 its in 15ms, 66 million/sec (24 cycles per loop)
    let kernel_src = r#"
        __kernel void add(
            __global float* c
        ) {
            float a = get_local_id(0);
            for (int i = 0; i < 1000000; i++) { a *= 2; }
            c[get_global_id(0)] = get_local_id(0);
            c[get_global_id(0)+128] = a;
        }
    "#;

    // Initialize ProQue (Program, Queue, Context)
    let proque = ProQue::builder()
        .src(kernel_src)
        .dims(1024)
        .build()?;

    let c_buffer = Buffer::builder().queue(proque.queue().clone()).flags(MemFlags::new().read_write()).len(32768).build()?;

    // Build kernel and set arguments
    let kernel = proque.kernel_builder("add")
        .arg(&c_buffer)
        .build()?;

    // warmup
    for _warmup in 0..2 {
        unsafe { kernel.cmd().global_work_size(1).local_work_size(1).enq()?; }
        proque.finish()?;
    }

    // draw plot
    let root = BitMapBackend::new("plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Based Line Plot", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..40960, 0..50000)?;

    chart.configure_mesh().draw()?;

    for (locals, color) in [(1, RED), (2, MAGENTA), (4, CYAN), (8, RED), (16, MAGENTA), (32, GREEN), (64, BLUE)] {
        let mut points = Vec::new();

        for test_cores in (64..min(40960, 640*2*locals)).step_by(max(64, locals)) {
            use std::time::Instant;
            let now = Instant::now();

            // Execute the kernel (unsafe due to GPU execution)
            // 256 is max threads on M3
            unsafe { kernel.cmd().global_work_size(test_cores).local_work_size(locals).enq()?; }

            // Wait for compute
            proque.finish()?;
            let elapsed = now.elapsed();
            println!("Elapsed: {:.2?} for {test_cores} with {locals}", elapsed);
            points.push((test_cores as i32, elapsed.as_micros() as i32));
        }
        chart.draw_series(LineSeries::new(points, &color))?;
    }

    // Read result back
    let mut c_data = vec![0.0f32; 128];
    c_buffer.cmd().read(&mut c_data).enq()?;

    // Verify output
    let mut i = 0;
    for &c in &c_data {
        if i % 16 == 0 && i != 0 {
            println!("");
        }
        i += 1;
        print!("{:>3} ", c);
        if i == 128 { break; }
        //assert_eq!(c, 3.0f32); // 1.0 + 2.0 = 3.0
    }
    println!("");

    Ok(())
}
