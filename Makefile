all:  LPproject


LPproject: LPproject.cu
	nvcc -o LPproject LPproject.cu

LPproject: LPproject.cu
	nvcc --expt-relaxed-constexpr -o LPproject LPproject.cu

clean:
	rm LPproject
