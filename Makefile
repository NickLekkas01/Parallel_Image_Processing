all: hostfile Paral_grey Paral_rgb Serial_grey Serial_rgb

hostfile:
	echo localhost slots=50 > hostfile

Paral_grey: Paral1.c
	mpicc Paral1.c -o Paral_grey -lm

Paral_rgb: Paral1.c
	mpicc Paral1.c -o Paral_rgb -lm -DUSE_RGB

Serial_grey: serial.c
	gcc serial.c -o Serial_grey

Serial_rgb: serial.c
	gcc serial.c -o Serial_rgb -DUSE_RGB
clean:
	rm -f hostfile Paral_grey Paral_rgb Serial_grey Serial_rgb
