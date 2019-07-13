TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python3.5/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0


g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/qwang014/.conda/envs/tensorflow-gpu/lib/python3.5/site-packages/tensorflow/include -I /home/qwang014/.conda/envs/tensorflow-gpu/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
