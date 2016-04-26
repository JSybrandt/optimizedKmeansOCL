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

 float dist(float4 p, float3 c){
	return sqrt(pow(p.x-c.x,2)+pow(p.y-c.y,2)+pow(p.z-c.z,2));
}

__kernel void Kmeans(__global float4* pA, __global float3* pB, __global float4* pC)
{
	const int CENTROID_COUNT = 3;
    const int x     = get_global_id(0);
    const int y     = get_global_id(1);
    const int width = get_global_size(0);
	float4 currPix;
    const int id = y * width + x;
	int i = 0;
	float minVal = 4*255;
	float3 c;
	float d;
	currPix = pA[id];


	for(i = 0 ; i < CENTROID_COUNT; i++){
		c =  pB[i];
		d = dist(currPix,c);
		if(minVal > d){
			minVal = d;
			currPix.w = i;
		 }
	}

	pC[id]=currPix;
	

}



