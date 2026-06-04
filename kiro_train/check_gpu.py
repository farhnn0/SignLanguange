"""Quick GPU check for TensorFlow"""
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("\nGPU devices:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"  ✓ {gpu.name}")
        print(f"    Device type: {gpu.device_type}")
else:
    print("  ✗ No GPU detected - will use CPU")

print("\nCUDA/cuDNN built with TensorFlow:")
print(f"  CUDA: {tf.test.is_built_with_cuda()}")
print(f"  GPU available: {tf.test.is_gpu_available(cuda_only=True) if hasattr(tf.test, 'is_gpu_available') else 'N/A (TF 2.1+)'}")

print("\nTesting GPU computation:")
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
    print(f"  ✓ GPU computation successful: {c.numpy()}")
except Exception as e:
    print(f"  ✗ GPU computation failed: {e}")
