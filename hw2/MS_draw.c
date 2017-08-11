 /* 
   Draw Mandelbort set
 */

#include <X11/Xlib.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
	char *MS_output=argv[1];
	FILE *fp = fopen(MS_output,"rb");
	int *arr = (int*)malloc(sizeof(int)*2);
	fread(arr,sizeof(int),2,fp);

	/* set window size */
	int width = arr[0];
	int height = arr[1];

    int *point = (int*)malloc(sizeof(int)*(width*height));
	fread(point,sizeof(int),width*height,fp);
	fclose(fp);

	Display *display;
	Window window;      //initialization for a window
	int screen;         //which screen 

	/* open connection with the server */ 
	display = XOpenDisplay(NULL);
	if(display == NULL) {
		fprintf(stderr, "cannot open display\n");
		return 0;
	}

	screen = DefaultScreen(display);

	/* set window position */
	int x = 0;
	int y = 0;

	/* border width in pixels */
	int border_width = 0;

	/* create window */
	window = XCreateSimpleWindow(display, RootWindow(display, screen), x, y, width, height, border_width,
					BlackPixel(display, screen), WhitePixel(display, screen));
	
	/* create graph */
	GC gc;
	XGCValues values;
	long valuemask = 0;
	
	gc = XCreateGC(display, window, valuemask, &values);
	//XSetBackground (display, gc, WhitePixel (display, screen));
	XSetForeground (display, gc, BlackPixel (display, screen));
	XSetBackground(display, gc, 0X0000FF00);
	XSetLineAttributes (display, gc, 1, LineSolid, CapRound, JoinRound);
	
	/* map(show) the window */
	XMapWindow(display, window);
	XSync(display, 0);
	
	/* draw points */
	int i, j;
	for(i=0; i<width; i++) {
		for(j=0; j<height; j++) {
			XSetForeground (display, gc,  1024 * 1024 * (point[i*height+j] % 256));		
			XDrawPoint (display, window, gc, i, j);
		}
	}
	XFlush(display);
	sleep(5);
	return 0;
}
