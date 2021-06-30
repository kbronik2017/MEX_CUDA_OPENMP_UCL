#pragma once

extern "C" void ThreeDMultiplicationCUBlas(int numARows,
	int numAColumns, int numBRows,
	int numBColumns, int numCRows, int numCColumns,
	int batch_count,
	 float **A,
	 float **B,
	float **C,
	float alpha,
	float beta);