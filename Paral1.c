#include <math.h>
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

typedef struct{
    // chunk dimensions
    int width;
    int height;

    int read_width;
    int read_height;


    // coordinates of chunk
    int x;
    int y;

    int dim_x;
    int dim_y;

    int img_width;
    int img_height;

    int offset_x;
    int offset_y;

    int rank;

    MYDATA** in;
    MYDATA** out;
} chunk;

int split_chunks(int img_height, int img_width, int size, int* chunk_height, int* chunk_width){
    int chunk_size = size;
    int i = 2;
    
    while (chunk_size != 1 && pow(i, 2) <= size)
    {
        if(chunk_size%i == 0)
        {
            //9,8//2
            //9,8//3
            //8,9//2
            //8,9//3
            //9,9//2
            if((img_height > img_width || (img_height <= img_width && img_width % i != 0 )) && img_height % i == 0  ){
                img_height /= i;
                chunk_size /= i;               
            } else if(img_width % i == 0){
                img_width /= i;
                chunk_size /= i;               
            }else 
            {
                if(i==2) i++;
                else i += 2;
            }
        }
        else
        {
            if(i==2) i++;
            else i += 2;
        }
    }

    *chunk_height = img_height;
    *chunk_width = img_width;
    if(chunk_size != 1)
    {
        printf("Chunk size = %d\n",chunk_size);
        return -1;
    }
    else
    {
        return 0;
    }
}

chunk* init_chunk(int img_height, int img_width, int rank, int size, int s, float resize){
    int ch_width, ch_height, split_failed, i;
    split_failed = split_chunks(img_height*resize, img_width*resize, size, &ch_height, &ch_width);
    if(split_failed)
    {
        printf("The number of threads given could not split the matrix into appropriate blocks. Change number of threads or a different image size.\n");
        return NULL;
    }
    if(ch_width < s || ch_height < s)    
    {
        printf("The blocks' size is not enough to contain enough data for the convolution. Change number of threads or a different image size.\n");
        return NULL;
    }
    chunk *c = malloc(sizeof(chunk));
    if(c == NULL)
    {
        printf("Out of memmory. Malloc failed.\n");
        return NULL;
    }
    c->width = ch_width;
    c->height = ch_height;
    
    int p_width = img_width * resize / c->width;
    
    c-> x = rank % p_width;
    c-> y= rank / p_width;

    c->dim_x = p_width;
    c->dim_y = img_height * resize / c->height;

    c->offset_x = c-> x * c->width;
    c->offset_y = c-> y * c->height;

    c->img_width = img_width;
    c->img_height = img_height;

    c->rank = rank;

    MYDATA *fullin, *fullout;
    if(c->height <= img_height)
    {
        c->read_height = c->height;
    }
    else
    {
        c->read_height = img_height;
    }

    if(c->width <= img_width)
    {
        c->read_width = c->width;
    }
    else
    {
        c->read_width = img_width;
    }

    c->in = malloc(c->height*sizeof(MYDATA*));
    fullin = malloc(c->height*c->width*sizeof(MYDATA));
    c->out = malloc(c->height*sizeof(MYDATA*));
    fullout = malloc(c->height*c->width*sizeof(MYDATA));
    
    for(i = 0; i < c->height; ++i)
    {
        c->in[i] = &fullin[i*c->width];
        c->out[i] = &fullout[i*c->width];
    }
    
    return c;    
}

void Send_array_boundaries(chunk *c, int s)
{
    // 0: top
    // 1: down
    // 2: left
    // 3: right
    // 4: top left
    // 5: top right
    // 6: bottom left
    // 7: bottom right
    int i, j;
    MPI_Request requests[8][s + c->height];
    int executed[8][s + c->height];
    for(i=0;i<8;i++) 
    {
        for(j=0;j<(s+c->height);j++)
        {
            executed[i][j] = 0;
        }
    }

    
    if(c->y - 1 >= 0)
    {
        int type = 0;
        for(i = 0; i<s;i++)
        {
            MPI_Isend(c->in[i], sizeof(MYDATA) * c->width, MPI_CHAR, (c->y-1)*c->dim_x+c->x, i * 8 + type, MPI_COMM_WORLD, &requests[type][i]);
            executed[type][i] = 1;
        }
    }

    if(c->y + 1 < c->dim_y)
    {
        int type = 1;
        for(i = 0; i<s;i++)
        {
            MPI_Isend(c->in[c->height-s+i], sizeof(MYDATA) * c->width, MPI_CHAR, (c->y+1)*c->dim_x+c->x, i * 8 + type, MPI_COMM_WORLD, &requests[type][i]);
            executed[type][i] = 1;
        }
    }

    if(c->x - 1 >= 0)
    {
        int type = 2;
        for(i = 0; i<c->height;i++)
        {
            MPI_Isend(c->in[i], sizeof(MYDATA) * s, MPI_CHAR, c->y*c->dim_x+(c->x-1), i * 8 + type, MPI_COMM_WORLD, &requests[type][i]);
            executed[type][i] = 1;
        }
    }

    if(c->x + 1 < c->dim_x)
    {
        int type = 3;
        for(i = 0; i<c->height;i++)
        {
            MPI_Isend(c->in[i]+c->width-s, sizeof(MYDATA) * s, MPI_CHAR, c->y*c->dim_x+(c->x+1), i * 8 + type, MPI_COMM_WORLD, &requests[type][i]);
            executed[type][i] = 1;
        }
    }

    if(c->y - 1 >= 0 && c->x-1 >= 0)
    {
        int type = 4;
        for(i = 0; i<s;i++)
        {
            MPI_Isend(c->in[i], sizeof(MYDATA) * s, MPI_CHAR, (c->y-1)*c->dim_x+(c->x-1), i * 8 + type, MPI_COMM_WORLD, &requests[type][i]);
            executed[type][i] = 1;
        }
    }

    if(c->y - 1 >= 0 && c->x+1 < c->dim_x)
    {
        int type = 5;
        for(i = 0; i<s;i++)
        {
            MPI_Isend(c->in[i]+c->width - s, sizeof(MYDATA) * s, MPI_CHAR, (c->y-1)*c->dim_x+(c->x+1), i * 8 + type, MPI_COMM_WORLD, &requests[type][i]);
            executed[type][i] = 1;
        }
    }

    if(c->y + 1 < c->dim_y && c->x-1 >= 0)
    {
        int type = 6;
        for(i = 0; i<s;i++)
        {
            MPI_Isend(c->in[c->height-s+i], sizeof(MYDATA) * s, MPI_CHAR, (c->y+1)*c->dim_x+(c->x-1), i * 8 + type, MPI_COMM_WORLD, &requests[type][i]);
            executed[type][i] = 1;
        }
    }

    if(c->y + 1 < c->dim_y && c->x+1 < c->dim_x)
    {
        int type = 7;
        for(i = 0; i<s;i++)
        {
            MPI_Isend(c->in[c->height-s+i]+c->width - s, sizeof(MYDATA) * s, MPI_CHAR, (c->y+1)*c->dim_x+(c->x+1), i * 8 + type, MPI_COMM_WORLD, &requests[type][i]);
            executed[type][i] = 1;
        }
    }

    for(i=0; i<8;i++)
    {
        for(j = 0;j<(s+c->height);j++)
        {
            if(executed[i][j]) 
            {
                MPI_Wait(&requests[i][j], MPI_STATUS_IGNORE);
            }
        }
    }
}

void Receive_array_boundaries(chunk *c, int s, MYDATA **UpperTable, MYDATA **LowerTable, MYDATA **LeftTable, MYDATA **RightTable)
{
    // 0: from down
    // 1: from up
    // 2: from right
    // 3: from left
    // 4: from bottom right
    // 5: from bottom left
    // 6: from top right
    // 7: from top left
    int i, j;
    MPI_Request requests[8][s + c->height];
    int executed[8][s + c->height];
    for(i=0;i<8;i++) 
    {
        for(j=0;j<(s+c->height);j++)
        {
            executed[i][j] = 0;
        }
    }
    if(c->y - 1 >= 0)
    {
        int type = 1;
        for(i = 0; i<s;i++)
        {
            MPI_Irecv(UpperTable[i], sizeof(MYDATA) * c->width, MPI_CHAR, (c->y-1)*c->dim_x+c->x, i * 8 + type, MPI_COMM_WORLD, &requests[type][i]);
            executed[type][i] = 1;
        }
    }

    if(c->y + 1 < c->dim_y)
    {
        int type = 0;
        for(i = 0; i<s;i++)
        {
            MPI_Irecv(LowerTable[i], sizeof(MYDATA) * c->width, MPI_CHAR, (c->y+1)*c->dim_x+c->x, i * 8 + type, MPI_COMM_WORLD, &requests[type][i]);
            executed[type][i] = 1;
        }
    }

    if(c->x - 1 >= 0)
    {
        int type = 3;
        for(i = 0; i<c->height;i++)
        {
            MPI_Irecv(LeftTable[s+i], sizeof(MYDATA) * s, MPI_CHAR, c->y*c->dim_x+(c->x-1), i * 8 + type, MPI_COMM_WORLD, &requests[type][i]);
            executed[type][i] = 1;
        }
    }

    if(c->x + 1 < c->dim_x)
    {
        int type = 2;
        for(i = 0; i<c->height;i++)
        {
            MPI_Irecv(RightTable[s+i], sizeof(MYDATA) * s, MPI_CHAR, c->y*c->dim_x+(c->x+1), i * 8 + type, MPI_COMM_WORLD, &requests[type][i]);
            executed[type][i] = 1;
        }
    }

    if(c->y - 1 >= 0 && c->x-1 >= 0)
    {
        int type = 7;
        for(i = 0; i<s;i++)
        {
            MPI_Irecv(LeftTable[i], sizeof(MYDATA) * s, MPI_CHAR, (c->y-1)*c->dim_x+(c->x-1), i * 8 + type, MPI_COMM_WORLD, &requests[type][i]);
            executed[type][i] = 1;
        }
    }

    if(c->y - 1 >= 0 && c->x+1 < c->dim_x)
    {
        int type = 6;
        for(i = 0; i<s;i++)
        {
            MPI_Irecv(RightTable[i], sizeof(MYDATA) * s, MPI_CHAR, (c->y-1)*c->dim_x+(c->x+1), i * 8 + type, MPI_COMM_WORLD, &requests[type][i]);
            executed[type][i] = 1;
        }
    }

    if(c->y + 1 < c->dim_y && c->x-1 >= 0)
    {
        int type = 5;
        for(i = 0; i<s;i++)
        {
            MPI_Irecv(LeftTable[s+c->height+i], sizeof(MYDATA) * s, MPI_CHAR, (c->y+1)*c->dim_x+(c->x-1), i * 8 + type, MPI_COMM_WORLD, &requests[type][i]);
            executed[type][i] = 1;
        }
    }

    if(c->y + 1 < c->dim_y && c->x+1 < c->dim_x)
    {
        int type = 4;
        for(i = 0; i<s;i++)
        {
            MPI_Irecv(RightTable[s+c->height+i], sizeof(MYDATA) * s, MPI_CHAR, (c->y+1)*c->dim_x+(c->x+1), i * 8 + type, MPI_COMM_WORLD, &requests[type][i]);
            executed[type][i] = 1;
        }
    }

    for(i=0; i<8;i++)
    {
        for(j = 0;j<(s+c->height);j++)
        {
            if(executed[i][j]) 
            {
                MPI_Wait(&requests[i][j], MPI_STATUS_IGNORE);
            }
        }
    }
}



void print_to_file(MYDATA **data, char *outname, int rank, int height, int width)
{        
    FILE *outfile;
    char filename[100];
    sprintf(filename, "%s%d.data", outname, rank);
    outfile = fopen(filename,"w");
    int myi,myj;
    for (myi = 0; myi < height; myi++)
    {
        for (myj = 0; myj < width; myj++)
        {
 
            fwrite(&data[myi][myj].color[0],1, 1, outfile);
        }
        //fprintf(outfile,"\n");
    }
    fclose(outfile);
}

void convolute(chunk* chunk, int s, float** h, MYDATA **UpperTable, MYDATA **LowerTable, MYDATA **LeftTable, MYDATA **RightTable, int part) 
{
    int offset, line_offset;
    int passed_tests = 1;
    int i,j, p, q;    
    #pragma omp parallel for shared(chunk, h, LeftTable, RightTable, LowerTable, UpperTable) private(i, j, p , q)
    for (i = 0; i < chunk->height; ++i)
    {            
        for (j = 0; j < chunk->width; ++j)
        {
            double result = 0;
            for (p = -s; p <= s; ++p)
            {         
                for (q = -s; q <= s; ++q)
                {
                    passed_tests = 1;
                    if((j-q) <0 && chunk->x-1 < 0) passed_tests = 0;
                    if((j-q) >= chunk->width && chunk->x+1 >= chunk->dim_x) passed_tests = 0;
                    if((i-p) < 0 && chunk->y-1 < 0) passed_tests = 0;
                    if((i-p) >= chunk->height &&  chunk->y+1 >= chunk->dim_y) passed_tests = 0;

                    if(passed_tests)
                    {
                        if((j-q)<0)
                        {
                            result += LeftTable[i-p+s][j-q+s].color[part] * h[p+s][q+s];    
                        }
                        else if((j-q)>=chunk->width)
                        {
                            result += RightTable[i-p+s][j - q - chunk->width].color[part] * h[p+s][q+s];    
                        }
                        else if((i-p) < 0)
                        {
                            result += UpperTable[i-p+s][j-q].color[part] * h[p+s][q+s];    
                        }
                        else if((i-p) >= chunk->height)
                        {
                            result += LowerTable[i-p-chunk->height][j-q].color[part] * h[p+s][q+s];    
                        }
                        else
                        {
                            result += chunk->in[i-p][j-q].color[part] * h[p+s][q+s];    
                        }
                    }
                    else
                    {
                        result += chunk->in[i][j].color[part] * h[p+s][q+s];
                    }    
                }
            }
            chunk->out[i][j].color[part] = result;
        }
    }
}


int main(int argc, char *argv[])
{
    struct timeval start,end;
    double cpu_time_used;
    char *outfilename = "final.data";
    MPI_Status stat;
    MPI_Init(&argc,&argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int failure = 0;
    if(rank == 0)
    {
        FILE *tempF = fopen(outfilename, "r");
        if(tempF != NULL)
        {
            failure = 1;
            fclose(tempF);
        }
        if(failure)
        {
            failure = remove(outfilename);
        }
        if(failure)
        {
            printf("File %s exists and could not be removed. Please delete it before running.\n", outfilename);
            fflush(stdout);
        }
    }
    MPI_Bcast(&failure, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if(failure)
    {
        MPI_Finalize();
        return 0;
    }
    int s;
    int width, height;
    char option[4];
    double resize = 1;
    int num_of_conv = 1;
    if (argc >= 3)
    {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        num_of_conv = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        resize = atof(argv[4]);
    }

    ssize_t datasizetype, datasizetypepointer;
    long int k = 0;
    MYDATA **Picture, **Picture2;
    MPI_File input;
    MPI_Status status;
    int i, j;
    float **h, *hfull;
    
    FILE *read;
    // printf("Give me the s \n");
    if(rank == 0) 
    {
        read = fopen("read.txt","r+");
        printf("I am rank %d and i closed input file\n", rank);
        fscanf(read, "%d", &s);
    }
    MPI_Bcast(&s, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("I am rank %d and got s = %d\n", rank,s);
    fflush(stdout);
    
    h = malloc((2*s+1) * sizeof(float *));
    hfull = malloc((2*s+1) * (2*s+1) * sizeof(float));
    
    for (i = 0; i < (2*s + 1); ++i)
        h[i] = &hfull[i*(2*s+1)];

    float sum = 0;
    //printf("Read the h \n");
    
    if(rank == 0)
    {
        for (i = 0; i < (2*s+1); ++i)
        {
            for (j = 0; j < (2*s+1); ++j)
            {
                int reader;
                fscanf(read, "%d", &reader);
                h[i][j] = (float)reader;
                sum += h[i][j];
            }
        }
        /*Divise every item with sum */
        for (i = 0; i < (2*s + 1); ++i)
            for (j = 0; j < (2*s + 1); ++j)
                h[i][j] /= sum;

    }

    MPI_Bcast(h[0], (2*s+1)*(2*s+1), MPI_FLOAT, 0, MPI_COMM_WORLD);

    printf("I am rank %d and got h[0][0] = %lf\n", rank,h[0][0]);
    
    if(rank == 0)
    {
        fclose(read);
        printf("I am rank %d and i closed input file\n", rank);
    }
    fflush(stdout);

    int new_height = height*resize;
    int new_width = width*resize;
    /*Create the chunk*/
    chunk *c = init_chunk(height, width, rank, size, s, resize);
    if(c == NULL) 
    {
        fflush(stdout);
        return 1;
    }
    printf("I am rank %d and init chunk\n", rank);
    fflush(stdout);

    int line_offset;
    //printf("Rank %d, X %d, Y %d, DIMX %d, DIMY %d, Resize %lf, OffsetX %d, OffsetY %d, ImgWidth %d, ImgHeight %d, Height %d, Width %d\n", 
    //rank, c->x, c->y, c->dim_x, c->dim_y,resize, c->offset_x, c->offset_y, c->img_width, c->img_height, c->height, c->width);
 
    MYDATA **UpperTable, **LowerTable, **LeftTable, **RightTable;
    MYDATA *UpperTablefull, *LowerTablefull, *LeftTablefull, *RightTablefull;

    UpperTable = malloc(s * sizeof(MYDATA*));
    UpperTablefull = malloc(s * c->width * sizeof(MYDATA));

    LowerTable = malloc(s * sizeof(MYDATA*));
    LowerTablefull = malloc(s * c->width * sizeof(MYDATA));

    for(i = 0; i < s; ++i)
    {
        UpperTable[i] = &UpperTablefull[i * c->width];
        LowerTable[i] = &LowerTablefull[i * c->width];
    }
    LeftTable = malloc((c->height + (2 * s)) * sizeof(MYDATA*));
    LeftTablefull = malloc((c->height +(2*s)) * s * sizeof(MYDATA));

    RightTable = malloc((c->height + 2 *s) * sizeof(MYDATA*));
    RightTablefull = malloc((c->height +2*s) * s * sizeof(MYDATA));
    for(i = 0; i < (c->height +2*s); ++i)
    {
        LeftTable[i] = &LeftTablefull[i * s];
        RightTable[i] = &RightTablefull[i * s];
    }
    
    // printf("I am rank %d and i reached here with offsetX %d offsetY %d datasize %d width %d height %d\n", rank, c->offset_x, c->offset_y, sizeof(MYDATA), c->read_width, c->read_height);
    // fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Finalize();
    // return 0;
    
#ifndef USE_RGB
    MPI_File_open(MPI_COMM_WORLD, "waterfall_grey_1920_2520.raw", MPI_MODE_RDONLY, MPI_INFO_NULL, &input);
#else
    MPI_File_open(MPI_COMM_WORLD, "waterfall_1920_2520.raw", MPI_MODE_RDONLY, MPI_INFO_NULL, &input);
#endif
    /*Sizeof(MYDATA) shows if we have RGB(3 char) or Grey scale(1 char) */
    /*c->offset_y*c->img_width + c->offset_x : Takes us to the first pixel of the current chunk */
    /*i*c->img_width : Move into every line at the current chunk */
    for(i = 0; i < c->read_height; ++i)
    {
        MPI_Status status;
        int offset = sizeof(MYDATA)*(((c->offset_y + i)%c->img_height )*c->img_width + (c->offset_x%c->img_width));
        int read_size = (c->img_width - (c->offset_x % c->img_width));
        if(read_size > c->read_width) read_size = c->read_width;
        MPI_File_read_at(input, offset, c->in[i], sizeof(MYDATA)*read_size, MPI_CHAR, &status);
        if(read_size < c->read_width)
        {
            offset = sizeof(MYDATA)*(((c->offset_y + i)%c->img_height )*c->img_width);
            MPI_File_read_at(input, offset, c->in[i]+read_size, sizeof(MYDATA)*(c->read_width - read_size), MPI_CHAR, MPI_STATUS_IGNORE);        
        }
    }
    
    MPI_File_close(&input);
    for ( i = 0; i<c->read_height;i++ )
    {
        for(j = c->read_width; j < c->width; j+= c->read_width)
        {
            int copy_size;
            if(c->width - j >= c->read_width) copy_size = c->read_width;
            else copy_size = c->width - j;
            memcpy(c->in[i]+j,c->in[i], sizeof(MYDATA)*copy_size);
        }
    }

    for(i = c->read_height; i< c->height; i++) memcpy(c->in[i],c->in[i%c->read_height], sizeof(MYDATA)*c->width);
    
    MPI_Barrier(MPI_COMM_WORLD);
        
    int p, q;
    gettimeofday(&start, NULL);
    
    //MPI_Type_vector(height/size, width, 0, MPI_CHAR, &MyDataType);
    for(i = 0; i < num_of_conv; i++)
    {
        if(i != 0)
        {
            memcpy(c->in[0], c->out[0], sizeof(MYDATA)*c->width * c->height);
        }

        Send_array_boundaries(c, s);

        Receive_array_boundaries(c, s, UpperTable, LowerTable, LeftTable, RightTable);
        
        convolute(c, s, h, UpperTable, LowerTable, LeftTable, RightTable, 0);
        
    #ifdef USE_RGB
        convolute(c, s, h, UpperTable, LowerTable, LeftTable, RightTable, 1);
        convolute(c, s, h, UpperTable, LowerTable, LeftTable, RightTable, 2);
    #endif
    gettimeofday(&end, NULL);
    

    }

    MPI_File output;
    printf("Rank %d, Resize %lf, OffsetX %d, OffsetY %d, ImgWidth %d, ImgHeight %d, Height %d, Width %d\n", 
    rank,resize, c->offset_x, c->offset_y, c->img_width, c->img_height, c->height, c->width);
    MPI_File_open(MPI_COMM_WORLD, outfilename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &output);
    for (i = 0; i < c->height; ++i)
    {
        int offset = sizeof(MYDATA)* ((c->offset_y + i) * (c->img_width * resize) + c->offset_x);
        MPI_File_write_at(output, offset ,c->out[i], sizeof(MYDATA)*(c->width), MPI_CHAR, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&output);

    // print_to_file(UpperTable, "upper", rank, s, c->width);
    // print_to_file(LowerTable, "lower", rank, s, c->width);
    // print_to_file(LeftTable, "left", rank, 2*s+c->height, s);
    // print_to_file(RightTable, "right", rank, 2*s+c->height, s);
    // print_to_file(c->in, "center", rank, c->height, c->width);

    free(UpperTable);
    free(UpperTablefull);
    free(LowerTable);
    free(LowerTablefull);
    free(LeftTable);
    free(LeftTablefull);
    free(RightTable);
    free(RightTablefull);

    free(c->in[0]);    
    free(c->in);
    free(c->out[0]);
    free(c->out);

    free(h[0]);
    free(h);


    
    cpu_time_used = ((end.tv_usec + end.tv_sec * 1000000) - (start.tv_usec + start.tv_sec * 1000000))/1000000.0;
    FILE *kp;
    kp = fopen("data.txt","w+");
    fprintf(kp,"CPU TIME: %f\n",cpu_time_used);
    fclose(kp);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;  

}

