#pragma once

extern "C" void  ElementwiseAddition(float * A, float * B, float * C, int numARows,
	int numAColumns,int numBRows, int numBColumns, int numCRows, int numCColumns, float alpha, float beta);