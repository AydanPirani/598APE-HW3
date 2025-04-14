# final location -309.392091 -242.794424
./main.exe 512 1024

# final location -8271867.836130 -4953006.283121
./main.exe 8 4194304

hyperfine "./main.exe 512 1024"
hyperfine "./main.exe 8 4194304"