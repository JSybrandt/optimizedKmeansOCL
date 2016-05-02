/*****************************************************************************
 * Copyright (c) 2013-2016 Intel Corporation
 * All rights reserved.
 *
 * WARRANTY DISCLAIMER
 *
 * THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
 * MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Intel Corporation is the author of the Materials, and requests that all
 * problem reports or change requests be submitted to it directly
 *****************************************************************************/

 float dist(float3 p, float3 c){
	return sqrt(pow(p.x-c.x,2)+pow(p.y-c.y,2)+pow(p.z-c.z,2));
}

//pixels x-r y-g z-b w-label
//centroid x-r y-g z-b
//centroidScratch index-centroid x-r y-b z-g w-count 
__kernel void Kmeans(__global float3* pixels, __global float4* centroidScratch, __global float3* output, __global float3* GLOBALcentroids)
{
	const int CENTROID_COUNT = 5;
    const int x     = get_global_id(0);
    const int y     = get_global_id(1);
    const int width = get_global_size(0);
	const int height = get_global_size(1);
	const int numThreads = width*height;
    const int id = y * width + x;

	float3 currPix = pixels[id]; 
	int label = -1;

	int scratchId = id*CENTROID_COUNT;
	
	bool conv = true;

	__local float3 centroids[5];
	if(x<CENTROID_COUNT)
		centroids[x] = GLOBALcentroids[x];

	int blah = 100;
	do{

		//LABEL
		float minVal = 9999;
		int newLabel = -1;
		int i = 0;
		for(i; i < CENTROID_COUNT; i++){
			float currDist = dist(currPix,centroids[i]);
			if(minVal > currDist){
				minVal = currDist;
				newLabel = i;
			 }
		}

		if(newLabel != label){
			conv = false;
		}
		label = newLabel;

		//prepare scratch
		for(i=0;i<CENTROID_COUNT;i++){
			centroidScratch[scratchId+i] = (float4)(0.0f,0.0f,0.0f,0.0f);
		}
		float4 scratchVal;
		scratchVal.x = currPix.x;
		scratchVal.y = currPix.y;
		scratchVal.z = currPix.z;
		scratchVal.w = 1; //count
		centroidScratch[scratchId+label] = scratchVal;

		//reduce
		
		int halfSize = numThreads>>1; //rounded up
		while(halfSize>0){
			barrier(CLK_GLOBAL_MEM_FENCE);
			if(id<halfSize){
				for(i=0;i<CENTROID_COUNT;i++){
					centroidScratch[scratchId+i] += centroidScratch[(id+halfSize)*CENTROID_COUNT+i]; //terrible //TODO:DOUBLE CHECK
				}
			}
			halfSize >>=1;
		}

		if(id==0){
			for(i=0;i<CENTROID_COUNT;i++){
				float4 temp = centroidScratch[i];
				temp.x /= temp.w;
				temp.y /= temp.w;
				temp.z /= temp.w;
				centroidScratch[i] = temp;
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);


		//load moved centroids to local
		for(i=0;i<CENTROID_COUNT;i++){
			centroids[i].x = centroidScratch[i].x; //terrible
			centroids[i].y = centroidScratch[i].y; //terrible
			centroids[i].z = centroidScratch[i].z; //terrible
		}
		blah--;
	}while(blah>0);
	output[id]=centroids[label];
}



