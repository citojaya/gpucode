#include <math.h>
#include <time.h>
#include "heydem.h"
#include "double_math.h"



int ParticleFeed(FeedZone *Feedcube, CylinderBC *h_Cir, double InitParSize, double3 *hPos, 
	             double  *hRad,   int numParticle,   int NP)
{

	double zMaxFeed = Feedcube->Lmax;
	double zMinFeed = Feedcube->Lmin;

	int nEach = 0;
	int JudgeVar = 0;
	srand(static_cast<unsigned int>(time(NULL)));

LOOP:
	while(JudgeVar < 500 && numParticle < NP && nEach < 60000)                  
	{

		double3 pos = make_double3(0.0, 0.0, 0.0);
		
		double theta = 2.0 * PI * static_cast<double>(rand())/RAND_MAX;
		double u = static_cast<double>(rand())/RAND_MAX + static_cast<double>(rand())/RAND_MAX;
		
		
		double mag = 0.0;
		if (u > 1) {
			mag = 2 - u;
		}
		else{
			mag = u;
		}
		pos.x = mag * (h_Cir->R - hRad[numParticle]) * cos(theta) + h_Cir->cir.x;
		pos.y = mag * (h_Cir->R - hRad[numParticle]) * sin(theta) + h_Cir->cir.y;
		
		
		double height = static_cast<double>(rand())/RAND_MAX;
		pos.z = height * (zMaxFeed - zMinFeed) + zMinFeed;



		// check overlap
		if (numParticle != 0)
		{
			for (int i=0; i<numParticle; i++)   
			{
				double3 ijPos = pos - hPos[i]; 
				double len = length(ijPos);
				if (len < (hRad[i] + hRad[numParticle] + 0.002))
				{
					JudgeVar = JudgeVar + 1;
					goto LOOP;
				}
			}
		}


		//*--- new particle position ---*//
		hPos[numParticle] = pos;

		numParticle = numParticle + 1;
		nEach = nEach + 1;
		JudgeVar = 0;

	}       

	
	return numParticle;
}
