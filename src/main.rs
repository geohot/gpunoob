extern crate vulkano;

use vulkano::device::{Device, DeviceExtensions};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::ComputePipeline;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::sync::GpuFuture;

// Custom error handling
#[derive(Debug)]
enum GpuError {
    NoSuitableDevice,
    PipelineCreationError,
    BufferCreationError,
}

// Enhanced configuration
struct GpuConfig {
    work_group_size: [u32; 3],
    data_size: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        GpuConfig {
            work_group_size: [1024, 1, 1],
            data_size: 1024 * 1024, // 1 million elements
        }
    }
}

fn main() -> Result<(), GpuError> {
    let config = GpuConfig::default();
    
    // Initialize Vulkan
    let instance = Instance::new(None, &vulkano::instance::InstanceExtensions::none(), None)
        .expect("Failed to create instance");

    // Device selection with error handling
    let physical = PhysicalDevice::enumerate(&instance)
        .next().ok_or(GpuError::NoSuitableDevice)?;

    let queue_family = physical.queue_families()
        .find(|&q| q.supports_compute())
        .expect("No compute queue family found");

    // Device creation
    let (device, mut queues) = Device::new(physical, &physical.supported_features(),
        &DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::none()
        }, [(queue_family, 0.5)].iter().cloned())
        .expect("Failed to create device");

    let queue = queues.next().unwrap();

    // Create buffers with better error handling
    let source_data = (0..config.data_size).map(|i| i as f32).collect::<Vec<_>>();
    
    let input_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        source_data.iter().cloned()
    ).map_err(|_| GpuError::BufferCreationError)?;

    let output_buffer = CpuAccessibleBuffer::<[f32]>::array(
        device.clone(),
        config.data_size as u64,
        BufferUsage::all(),
        false
    ).map_err(|_| GpuError::BufferCreationError)?;

    // Enhanced shader loading
    let shader = cs::Shader::load(device.clone())
        .map_err(|_| GpuError::PipelineCreationError)?;

    // Pipeline creation
    let pipeline = Arc::new(ComputePipeline::new(
        device.clone(),
        &shader.main_entry_point(),
        &()
    ).map_err(|_| GpuError::PipelineCreationError)?);

    // Descriptor set
    let set = Arc::new(
        PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_buffer(input_buffer.clone()).unwrap()
            .add_buffer(output_buffer.clone()).unwrap()
            .build().unwrap()
    );

    // Command buffer building
    let mut builder = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
    builder.dispatch(
        [config.data_size as u32 / config.work_group_size[0], 1, 1],
        pipeline.clone(),
        set.clone(),
        (),
    ).unwrap();
    let command_buffer = builder.build().unwrap();

    // Execution and timing
    let start = std::time::Instant::now();
    let future = command_buffer.execute(queue.clone())
        .then_signal_fence_and_flush()
        .unwrap();
    
    future.wait(None).unwrap();
    let elapsed = start.elapsed();

    // Verify results
    let input_content = input_buffer.read().unwrap();
    let output_content = output_buffer.read().unwrap();
    
    for (input, output) in input_content.iter().zip(output_content.iter()) {
        assert_eq!(*output, input * 2.0); // Verify our shader math
    }

    println!("Computation completed in {:?}", elapsed);
    println!("First 10 results: {:?}", &output_content[0..10]);

    Ok(())
}

// Enhanced shader module
mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
            #version 450

            layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Data {
                float[] input_data;
                float[] output_data;
            };

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                output_data[idx] = input_data[idx] * 2.0; // Changed from addition to multiplication
            }
        "
    }
}
