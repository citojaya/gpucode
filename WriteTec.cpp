#include <fstream>
#include <iomanip>
#include "heydem.h"
#include "double_math.h"




void WrtTec(double3 *hPos,		double3 *hVel, 
			double3 *hForce,	double3 *hAngVel,
	        SIMULATION *h_dem,	double *hRad, 
			UNIT *Rdu,			int TotalParticle,	int NP)
{
	FILE *outfile; 
	char filename[20];
	sprintf(filename, "particle_info.dat");

	if (h_dem->Outs == 0)
	{
		outfile = fopen(filename, "wt");

		fprintf(outfile, "TITLE = \" PARTICLE INFORMATION \" \n");
		fprintf(outfile, "VARIABLES = X   Y   Z   R   VX   VY   VZ   W   F\n");
		fclose(outfile);
	}

	outfile = fopen(filename, "a");
	fprintf(outfile, "ZONE T= \" %12.6lf s \" \n", h_dem->ctime * Rdu->rtunit);
	for (int ip = 0; ip<TotalParticle; ip++)
	{
		fprintf(outfile, "%11.5lf   %11.5lf   %11.5lf   %11.5f  %11.5lf   %11.5lf   %11.5lf   %11.5lf   %11.5lf\n", 
		                  hPos[ip].x,   hPos[ip].y,   hPos[ip].z,  hRad[ip],
		                  hVel[ip].x,   hVel[ip].y,   hVel[ip].z, 
		                  length(hAngVel[ip]), length(hForce[ip]));
	}

	fclose(outfile);
}




