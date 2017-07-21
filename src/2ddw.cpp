#include "2ddw.h"
#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include <omp.h>
#include <string.h>


#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define INFINITY_DTW 10000000.f

inline int index2(const int x, const int y,  const int xSize, const int ySize ) {
    return x * ySize + y;
}

inline int index3(const int x, const int y, const int z, const int xSize, const int ySize, const int zSize ) {
    return x * ySize * zSize + y * zSize + z;
}

inline int index4(const int w, const int x, const int y, const int z, const int xwize, const int xSize, const int ySize, const int zSize ) {
    return w * xSize * ySize * zSize+ x * ySize * zSize + y * zSize + z;
}

inline int index5(const int v,const int w, const int x, const int y, const int z, const int vSize, const int wSize, const int xSize, const int ySize, const int zSize ) {
    return w * wSize * xSize * ySize * zSize + w * xSize * ySize * zSize+ x * ySize * zSize + y * zSize + z;
}

inline float d_bis(float a, float b){
    return fabs(b-a);
}


void calculateHalfCoefX(float *data1, float *data2, int sizeData1X, int sizeData1Y,int sizeData2X,int sizeData2Y, int y1, int y2, float *coefX, float *workSpace){


    float *dtwArray=workSpace;//(float*)malloc(size2*size1*sizeof(float));

    for (int i = 1; i < sizeData1X; ++i)
    {
        dtwArray[i] = INFINITY_DTW;
    }
    for (int i = 1; i < sizeData2X; ++i)
    {
        dtwArray[i*sizeData1Y] = INFINITY_DTW;
    }

    dtwArray[0] = 0.f;

    // calcule

    for (int x1 = 1; x1 < sizeData1X; ++x1)
    {
        for (int x2 = 1; x2 < sizeData2X; ++x2)
        {
            float cost= d_bis(data1[index2(x1,y1,sizeData1X,sizeData1Y)], data2[index2(x2,y2,sizeData2X,sizeData2Y)]);
            //fprintf(stderr, "%s %f\n", "cost =",cost);
            dtwArray[x1+x2*sizeData1Y] = cost + MIN(MIN(dtwArray[(x1-1)+x2*sizeData1Y ],    // insertion
                                                dtwArray[x1+(x2-1)*sizeData1Y]),    // deletion
                                                dtwArray[(x1-1)+(x2-1)*sizeData1Y]);    // match
        }
    }

    //remplissage
    //float valeur=dtwArray[sizeData1Y*sizeData2Y-1];
    for (int x1 = 1; x1 < sizeData1X; ++x1)
    {
        for (int x2 = 1; x2 < sizeData2X; ++x2)
        {
            coefX[index4(x1,y1,x2,y2,sizeData1X,  sizeData1Y, sizeData2X, sizeData2Y)]=dtwArray[y1+y2*sizeData1Y];
        }
    }
   

}

void calculateHalfCoefY(float *data1, float *data2, int sizeData1X, int sizeData1Y,int sizeData2X,int sizeData2Y, int x1, int x2, float *coefX, float *workSpace){


    float *dtwArray=workSpace;//(float*)malloc(size2*size1*sizeof(float));

    for (int i = 1; i < sizeData1Y; ++i)
    {
        dtwArray[i] = INFINITY_DTW;
    }
    for (int i = 1; i < sizeData2Y; ++i)
    {
        dtwArray[i*sizeData1Y] = INFINITY_DTW;
    }

    dtwArray[0] = 0.f;

    // calcule

    for (int y1 = 1; y1 < sizeData1Y; ++y1)
    {
        for (int y2 = 1; y2 < sizeData2Y; ++y2)
        {
            float cost= d_bis(data1[index2(x1,y1,sizeData1X,sizeData1Y)], data2[index2(x2,y2,sizeData2X,sizeData2Y)]);
            //fprintf(stderr, "%s %f\n", "cost =",cost);
            dtwArray[y1+y2*sizeData1Y] = cost + MIN(MIN(dtwArray[(y1-1)+y2*sizeData1Y ],    // insertion
                                                dtwArray[y1+(y2-1)*sizeData1Y]),    // deletion
                                                dtwArray[(y1-1)+(y2-1)*sizeData1Y]);    // match
        }
    }

    //remplissage
    //float valeur=dtwArray[sizeData1Y*sizeData2Y-1];
    for (int y1 = 1; y1 < sizeData1Y; ++y1)
    {
        for (int y2 = 1; y2 < sizeData2Y; ++y2)
        {
            coefX[index4(x1,y1,x2,y2,sizeData1X,  sizeData1Y, sizeData2X, sizeData2Y)]=dtwArray[y1+y2*sizeData1Y];
        }
    }
   

}

float BiDDW(float *data1, float *data2,float *data1_flip, float *data2_flip, int const sizeData1X, int const sizeData1Y,int const sizeData2X,int const sizeData2Y){

    float *coefX=(float*)malloc(sizeData1X*sizeData2X*sizeData1Y*sizeData2Y*sizeof(float));
    float *coefY=(float*)malloc(sizeData1X*sizeData2X*sizeData1Y*sizeData2Y*sizeof(float));

 
    float workSpaceUnique[MAX(sizeData1Y*sizeData2Y,sizeData1X*sizeData2X)];

    //float* workSpace[8]={workSpace0, workSpace1, workSpace2, workSpace3, workSpace4, workSpace5, workSpace6, workSpace7};
    omp_set_num_threads(8);
    //#pragma omp parallel 
    {
        #pragma omp parallel for default(none) private(workSpaceUnique) firstprivate(coefX,coefY,data1,  data2,  sizeData1X,  sizeData1Y, sizeData2X, sizeData2Y)
        for (int x1 = 0; x1 < sizeData1X; ++x1)
        {
           
            //#pragma omp parallel for private(workSpaceUnique)
            for (int x2 = 0; x2 < sizeData2X; ++x2)
            {
                calculateHalfCoefY( data1,  data2,  sizeData1X,  sizeData1Y, sizeData2X, sizeData2Y,  x1,  x2,  coefX, workSpaceUnique); 
            }
        }
       #pragma omp parallel for default(none) private(workSpaceUnique) firstprivate(coefX,coefY,data1,  data2,  sizeData1X,  sizeData1Y, sizeData2X, sizeData2Y)
        for (int y1 = 0; y1 < sizeData1Y; ++y1)
        {
      
            //#pragma omp parallel for private(workSpaceUnique)
            for (int y2 = 0; y2 < sizeData2Y; ++y2)
            {
                calculateHalfCoefX( data1,  data2,  sizeData1X,  sizeData1Y, sizeData2X, sizeData2Y,  y1,  y2,  coefY, workSpaceUnique); 
            }
        }
    }

    float *array=(float*)malloc(sizeData1X*sizeData1Y*sizeData2X*sizeData2Y*sizeof(float));
    memset(array,0,sizeData1X*sizeData1Y*sizeData2X*sizeData2Y*sizeof(float));
    
    for (int i = 0; i < sizeData1X*sizeData1Y*sizeData2X*sizeData2Y; ++i)
    {
        array[i]=INFINITY_DTW;
    }
    array[0]=0.f;

    int const x1Delta[16]={-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0};
    int const y1Delta[16]={-1,-1,-1,-1,0,0,0,0,-1,-1,-1,-1,0,0,0,0};
    int const x2Delta[16]={-1,-1,0,0,-1,-1,0,0,-1,-1,0,0,-1,-1,0,0};
    int const y2Delta[16]={-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0};
    //omp_set_num_threads(8);
    //memset(array,0,sizeData1X*sizeData1Y*sizeData2X*sizeData2Y*sizeof(float));
    #pragma omp parallel default(none) firstprivate(array,  sizeData1X,  sizeData1Y, sizeData2X, sizeData2Y, x1Delta ,y1Delta, x2Delta,y2Delta,coefX,coefY )
    {
        int fullSize=sizeData1X+sizeData1Y+sizeData2X+sizeData2Y;

        for (int r = 1; r < fullSize; ++r)
        {
            //fprintf(stderr, "r =%d\n",r);
            #pragma omp for 
            for (int x1 = MAX(1,r-sizeData2Y-sizeData2X-sizeData1Y); x1 < MIN(sizeData1X,r); ++x1)
            {
                for (int y1 = MAX(1,r-sizeData2Y-sizeData2X-x1); y1 < MIN(sizeData1Y,r-x1); ++y1)
                {
                    for (int x2 = MAX(1,r-sizeData2Y-x1-y1); x2 < MIN(sizeData2X,r-x1-y1); ++x2)
                    {
                        int y2=r-x1-y1-x2;
                        
                        //for (int y2 = MAX(1,x2-sizeData2X); y2 < MIN(sizeData2Y,r-x1-y1-x2); ++y2)
                        {
                            float valeur=INFINITY_DTW;
                            for (int k = 0; k < 16; ++k)
                            {
                                
                                int x1local=x1+x1Delta[k];
                                int y1local=y1+y1Delta[k];
                                int x2local=x2+x2Delta[k];
                                int y2local=y2+y2Delta[k];

                                float cost=0.f;
                                cost+=coefX[index4(x1local,y1local,x2local,y2local,sizeData1X,sizeData1Y,sizeData2X,sizeData2Y)];
                                cost+=coefY[index4(x1local,y1local,x2local,y2local,sizeData1X,sizeData1Y,sizeData2X,sizeData2Y)];
                                float tmpValeur= cost + array[index4(x1local,y1local,x2local,y2local,sizeData1X,  sizeData1Y, sizeData2X, sizeData2Y)];
                                if(tmpValeur<valeur)valeur=tmpValeur;
                            }
                            array[index4(x1,y1,x2,y2,sizeData1X, sizeData1Y, sizeData2X, sizeData2Y)]=valeur;
                        }
                    }
                }
            }
        }
    }

   /* int nombre0=0;
    int nombre1=0;
    int nombreAutre=0;
    for (int i = 0; i < (sizeData1X*sizeData1Y*sizeData2X*sizeData2Y); ++i)
    {
        float valeur=array[i];
        if(valeur==0)nombre0++;
        if(valeur==1)nombre1++;
        if(valeur>1)nombreAutre++;
    }*/

    //printf("nbO:%d, nb1:%d, nmAautre:%d\n",nombre0,nombre1,nombreAutre );

    //#pragma omp parallel
    /*{
        for (int x1 = 1; x1 < sizeData1X; ++x1)
        {
            for (int y1 = 1; y1 < sizeData1Y; ++y1)
            {
                for (int x2 = 1; x2 < sizeData2X; ++x2)
                {
                    for (int y2 = 1; y2 < sizeData2Y; ++y2)
                    {
                        float valeur=INFINITY_DTW;
                         #pragma omp parallel for  firstprivate(coefX,coefY,x1,x2,y1,y2,array) reduction(min : valeur) 
                        for (int k = 0; k < 16; ++k)
                        {
                            
                            int x1local=x1+x1Delta[k];
                            int y1local=y1+y1Delta[k];
                            int x2local=x2+x2Delta[k];
                            int y2local=y2+y2Delta[k];

                            float cost=0.f;
                            cost+=coefX[index4(x1local,y1local,x2local,y2local,sizeData1X,sizeData1Y,sizeData2X,sizeData2Y)];
                            cost+=coefY[index4(x1local,y1local,x2local,y2local,sizeData1X,sizeData1Y,sizeData2X,sizeData2Y)];
                            float tmpValeur= cost + array[index4(x1local,y1local,x2local,y2local,sizeData1X,  sizeData1Y, sizeData2X, sizeData2Y)];
                            if(tmpValeur<valeur)valeur=tmpValeur;
                        }
                        array[index4(x1,y1,x2,y2,sizeData1X, sizeData1Y, sizeData2X, sizeData2Y)]=valeur;
                    }
                }
            }
        }
    }*/
    float valeur=array[index4(sizeData1X-1,sizeData1Y-1,sizeData2X-1,sizeData2Y-1,sizeData1X,  sizeData1Y, sizeData2X, sizeData2Y)];
    free(coefX);
    free(coefY);

    /*for (int i = 0; i < 8; ++i)
    {
       free(workSpace[i]);
    }*/

    return valeur;
}