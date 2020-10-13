#/bin/bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_NSYNC=/home/marie-julie/tf/local/lib/python2.7/site-packages/tensorflow/include/external/nsync/public/
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#/usr/local/cuda-10.0/bin/nvcc -D_GLIBCXX_USE_CXX11_ABI=0 tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -I $TF_INC -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# TF1.2
#g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $TF_INC -I /usr/local/cuda-10.0/include -I $TF_NSYNC -lcudart -L /usr/local/cuda-10.0/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
#g++ -std=c++11 tf_nndistance_g.cpp tf_nndistance_g.cu.o -o tf_nndistance_g_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
#	/usr/local/cuda/bin/nvcc -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $TF_INC -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


/usr/local/cuda-10.0/bin/nvcc  -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $TF_INC -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 #-D_GLIBCXX_USE_CXX11_ABI=0
# #g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $TF_INC -lcudart -L /usr/local/cuda-10.0/lib64/ -L$TF_LIB -ltensorflow_framework -O2 #-D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $TF_INC -I /usr/local/cuda-10.0/include -I $TF_NSYNC -lcudart -L /usr/local/cuda-10.0/lib64/ -L$TF_LIB -l:libtensorflow_framework.so.1 -O2 -D_GLIBCXX_USE_CXX11_ABI=0  #

#/usr/local/cuda-10.0/bin/nvcc -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $TF_INC -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
#g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I$TF_INC -I$TF_NSYNC -L$TF_LIB -ltensorflow_framework -I/usr/local/cuda-10.0/include -lcudart -O2 -D_GLIBCXX_USE_CXX11_ABI=0
