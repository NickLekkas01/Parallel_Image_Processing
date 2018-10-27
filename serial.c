#include <mpi.h>
#include <omp.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>


#ifndef USE_RGB
#define MYDATA Greyscale
#else
#define MYDATA RGB
#endif

typedef struct rgb
{
    unsigned char color[3];
} RGB;

typedef struct greyscale
{
    unsigned char color[1];
} Greyscale;

void convolute(MYDATA **Picture, MYDATA **Picture2, int new_height, int new_width, int height, int width, float** h, int s, int part) {
    int i, j, p, q;
    #pragma omp parallel for shared(Picture, Picture2, h) private(i, j, p, q)
    for (i = 0; i < new_height; ++i)
    {
        for (j = 0; j < new_width; ++j)
        {
            double result = 0;
            Picture2[i][j].color[part] = 0;
            for (p = -s; p <= s; ++p)
            {         
                for (q = -s; q <= s; ++q)
                {
                    if (i - p < 0 || i - p >= new_height || j - q < 0 || (j-q) >= new_width)
                        result += Picture[i % height][j % width].color[part] * h[p+s][q+s];
                    else
                        result += Picture[(i-p) % height][(j-q) % width].color[part] * h[p+s][q+s];    
                }
            }
            Picture2[i][j].color[part] = result;
        }
    }

}


int main(int argc, char *argv[])
{
    struct timeval start,end;
    double cpu_time_used;
    
    float **h;
    int s;
    int width, height;
    char option[4];
    int i, j;
    double resize=1;
    int num_of_conv=1;
    if (argc == 3)
    {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }
    else if (argc >= 4)
    {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
        num_of_conv = atoi(argv[3]);
    }
    else if (argc >= 5)
    {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
        num_of_conv = atoi(argv[3]);
        resize = atof(argv[4]);
    }
    else
    {
        printf("Wrong command parameters\n");
        return -1;
    }

    fflush(stdout);
    FILE *read;
    read = fopen("read.txt","r+");
    // printf("Give me the s \n");
    fscanf(read, "%d", &s);
    h = malloc((2*s + 1) * sizeof(float *));
    for (int i = 0; i < (2*s + 1); ++i)
        h[i] = malloc((2*s + 1) * sizeof(float));

    float sum = 0;
    //printf("Read the h \n");
    for (int i = 0; i < (2*s + 1); ++i)
    {
        for (int j = 0; j < (2*s + 1); ++j)
        {
            int reader;
            fscanf(read, "%d", &reader);
            h[i][j] = (float)reader;
            sum += h[i][j];
        }
    }
    for (int i = 0; i < (2*s + 1); ++i)
        for (int j = 0; j < (2*s + 1); ++j)
            h[i][j] /= sum;

    ssize_t datasizetype, datasizetypepointer;
    long int k = 0;
    MYDATA **Picture, **Picture2;
    MYDATA *buffer;
    FILE *input;

#ifndef USE_RGB
    input = fopen("waterfall_grey_1920_2520.raw", "rb");
#else
    input = fopen("waterfall_1920_2520.raw", "rb");
#endif

    
    buffer = malloc((width * height ) * sizeof(MYDATA));

    while ((fread(&(buffer[k++]), sizeof(MYDATA), 1, input)) == 1)
        ;
    k--;
    
    
    fclose(input);
    
    int new_height = height*resize;
    int new_width = width*resize;
    

    MYDATA *fullPicture,*fullPicture2;
    Picture = malloc(height * sizeof(MYDATA *));  
    fullPicture = malloc(height*width*sizeof(MYDATA));
    Picture2 = malloc(new_height * sizeof(MYDATA *)); 
    fullPicture2 = malloc(new_height*new_width*sizeof(MYDATA));
    for (int i = 0; i < height; ++i)                         
    {
        Picture[i] = &fullPicture[i*width];  
    }
    for (int i = 0; i < new_height; ++i)                         
    {
        Picture2[i] = &fullPicture2[i*new_width];    
    }    
    int t = 0;
    for (i = 0; i < k; ++i)
    {
        Picture[t][i % width] = buffer[i];
        if ((i % width) == (width - 1))
            t++;
    }

    for(i = 0; i < num_of_conv; i++)
    {
        if(i != 0)
        {
            memcpy(Picture[0], Picture2[0], sizeof(MYDATA)*new_width * new_height);
        }
        
        convolute(Picture,Picture2,new_height, new_width, height, width, h, s, 0);    
    #ifdef USE_RGB
        convolute(Picture,Picture2,new_height, new_width, height, width, h, s, 1);    
        convolute(Picture,Picture2,new_height, new_width, height, width, h, s, 2);    
    #endif
    }
   
    

    for (int i = 0; i < new_height; ++i)
        for (int j = 0; j < new_width; ++j)
        {
            fwrite(&(Picture2[i][j]), sizeof(MYDATA), 1, stdout);
        }
        
    free(Picture[0]);
    free(Picture);
    free(Picture2[0]);
    free(Picture2);

    for (int i = 0; i < 2*s + 1; ++i)
        free(h[i]);
    free(h);

    
    cpu_time_used = ((end.tv_usec + end.tv_sec * 1000000) - (start.tv_usec + start.tv_sec * 1000000))/1000000.0;
    FILE *kp;
    kp = fopen("data.txt","w+");
    fprintf(kp,"CPU TIME: %f\n",cpu_time_used);
    fclose(kp);
    return 0;
}
