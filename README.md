# Parallelism-project  
`cpu.cpp ` `gpu.cu` это старые версии кода на cpu и gpu соответственно  
`openmp.cpp` это рабочая версия на openmp  
`gpu_1d_1worker.cu` ,базовая версия на cuda  
`gpu_1d_many_wo_sm.cu` 1D версия на cuda без shared memory но с несколькими потоками  
`gpu_1d_many_w_with_sh_mem.cu` 1D версия на cuda с shared memory и с несколькими потоками  
`2D.cu` 2D версия на cuda с shared memory и с несколькими потоками  
`mpi.cpp` версия на OpenMPI 


`acc.cu ` не особо рабочая версия на acc
