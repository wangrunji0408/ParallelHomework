#include <iostream>
#include <cstring>
#include <unistd.h>
#include <pthread.h>
#include <curses.h>
#include <cstdlib>
using namespace std;

const int MAXH = 10;
const int MAXW = 100;
const int FPS = 30;

enum Direction
{
	UP, DOWN, LEFT, RIGHT, NONE
};

enum Status
{
	NORMAL, WIN, LOSE
};

struct Lane
{
	bool v[MAXW];
	int width;
	Direction dir;
	int moveRound;
	int carLength, interval;
	int i;

	Lane(){}

	Lane (int width, Direction dir, int moveRound, int carLength, int interval, int i)
	{
		memset(v, 0, sizeof(v));
		this->width = width;
		this->dir = dir;
		this->moveRound = moveRound;
		this->carLength = carLength;
		this->interval = interval;
		this->i = i;
	}

	void next ()
	{
		bool gen = i < carLength * moveRound;
		i = i == interval * moveRound - 1? 0: i + 1;
		if(i % moveRound == 0)
		{
			if(dir == LEFT)
			{
				memcpy(v, v + 1, (width - 1) * sizeof(bool));
				v[width - 1] = gen;
			}
			else if(dir == RIGHT)
			{
				for(int j=width-1; j>0; --j)
					v[j] = v[j-1];
				v[0] = gen;
			}
		}
	}
};

class Game
{
public:
	// Map
	int height, width;
	// Lane info
	Lane lane[MAXH];
	// Frog info
	int y, x;
	Status status;

private:
	void moveFrog (Direction action)
	{
		switch(action)
		{
			case UP: y++; break;
			case DOWN: y--; break;
			case LEFT: x--; break;
			case RIGHT: x++; break;
			case NONE: break;
			default: break;
		}
		if(x < 0)	x = 0;
		if(x >= width) x = width - 1;
		if(y < 0)	y = 0;
		if(y > height + 1) y = height + 1;
	}
	Status checkStatus () const
	{
		if(y == height + 1)
			return WIN;
		if(y > 0 && lane[y].v[x] == true)
			return LOSE;
		return NORMAL;
	}
public:
	Game()
	{
		status = NORMAL;
		height = 4; width = 80;
		y = 0; x = width / 2;
		lane[1] = Lane(width, LEFT, 5, 3, 10, 0);
		lane[2] = Lane(width, RIGHT, 7, 2, 8, 2);
		lane[3] = Lane(width, LEFT, 8, 3, 6, 0);
		lane[4] = Lane(width, RIGHT, 10, 2, 7, 2);
		for(int i=0; i<10000; ++i)
			goNextFrame(NONE);
	}
	void restart ()
	{
		*this = Game();
	}
	void goNextFrame (Direction action)
	{
		for(int i=1; i<=height; ++i)
			lane[i].next();
		moveFrog(action);
		status = checkStatus();
	}
	void print ()
	{
		static char buffer[MAXH][MAXW];
		for(int i=height; i>=1; --i)
			for(int j=0; j<width; ++j)
				buffer[i][j] = lane[i].v[j]? 'x': ' ';
		for(int j=0; j<width; ++j)
			buffer[0][j] = ' ';
		for(int j=0; j<width; ++j)
			buffer[height+1][j] = ' ';
		buffer[y][x] = '.';

		clear();
		for(int i=height+1; i>=0; --i)
			mvaddstr(height+1 - i, 0, buffer[i]);
		if(status == WIN)
			mvaddstr(height+1, 0, "You WIN! Press 'R' to restart.");
		else if(status == LOSE)
			mvaddstr(height+1, 0, "OUCH!    Press 'R' to restart.");
		refresh();
	}
};

Direction action = NONE;
bool pressR;
void* getInput (void* a)
{
    cbreak();
    noecho();
	int ch;
	while(true)
	{
		ch = getch();
		if(ch == 3)		exit(1);
		if(ch == 'r')	{pressR = true; continue;}
		if(ch != 27)	continue;
		if(getch() != 91)	continue;
		switch(getch())
        {
			case 65:   action = UP; break;
			case 68:   action = LEFT; break;
			case 67:   action = RIGHT; break;
			case 66:   action = DOWN; break;
			default:   break;
        }
	}
}

void waitKeyR ()
{
	pressR = false;
	while(pressR == false)
		usleep(1000000 / FPS);
	pressR = false;
}

int main ()
{
	initscr();
	Game g;
	pthread_t thread;
	pthread_create(&thread, NULL, getInput, NULL);
	while(true)
	{
		action = NONE;
		usleep(1000000 / FPS);
		g.goNextFrame(action);
		g.print();
		if(g.status != NORMAL)
		{
			waitKeyR();
			g.restart();
		}
	}
	endwin();
	return 0;
}