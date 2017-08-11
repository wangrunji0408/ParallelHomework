 /* 
   Sequential Mandelbort set
 */

#include <stdio.h>
#include <stdlib.h>

typedef struct complextype
{
	double real, imag;
} Compl;

int main(int argc, char* argv[])
{
	int thread_num=atoi(argv[1]); 
	double  realaxis_left=atof(argv[2]);
	double realaxis_right=atof(argv[3]);
	double imageaxis_lower=atof(argv[4]);
	double imageaxis_upper=atof(argv[5]);
	int xpoint=atoi(argv[6]);
	int ypoint=atoi(argv[7]);
	char *output=argv[8];

	/* set window size */
	int width = xpoint;
	int height = ypoint;
	
	/* calculate points */
	Compl z, c;
	int repeats;
	double temp, lengthsq;
	int i, j;
	int *BUF=malloc((width*height+2)*sizeof(int));
	BUF[0] = width;
	BUF[1] = height;
	
	for(i=0; i<width; i++) {
		for(j=0; j<height; j++) {
			z.real = 0.0;
			z.imag = 0.0;
			c.real = realaxis_left + (double)i * ((realaxis_right-realaxis_left)/(double)width); /* Theorem : If c belongs to M(Mandelbrot set), then |c| <= 2 */
			c.imag = imageaxis_lower + (double)j * ((imageaxis_upper-imageaxis_lower)/(double)height); /* So needs to scale the window */
			repeats = 0;
			lengthsq = 0.0;

			while(repeats < 100000 && lengthsq <= 4.0) { /* Theorem : If c belongs to M, then |Zn| <= 2. So Zn^2 <= 4 */
				temp = z.real*z.real - z.imag*z.imag + c.real;
				z.imag = 2*z.real*z.imag + c.imag;
				z.real = temp;
				lengthsq = z.real*z.real + z.imag*z.imag; 
				repeats++;
			}
			BUF[i*height+j+2]=repeats;
		}
	}
	FILE *fp;
	fp = fopen(output,"wb");
	fwrite(BUF, sizeof(int), width*height+2 , fp);
	fclose(fp);
	return 0;
}
