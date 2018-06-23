# PdeFiniteDifferenceSolver
C++ manager class for PdeFiniteDifferenceKernels API. The low level calls are managed in the namespace <i>pde::detail DeviceManager</i>, whereas the high level infrastructure is delegated to the particular solver type. 

Only linear hyperbolic and parabolic PDEs are supported (up to 3D). The exposed implementation is through:
- <i>AdvectionDiffusionSolver1D, AdvectionDiffusionSolver2D</i>
- <i>WaveEquationSolver1D, WaveEquationSolver2D</i>

These solvers are implemented with the Curiously Recurring Template Pattern (CRTP), useful for delegating the members data type at compile time.

For convenience's sake the following typedefs have been defined:

- Advection-Diffusion:
```c++
	typedef AdvectionDiffusionSolver1D<MemorySpace::Device, MathDomain::Float> GpuSingleAdvectionDiffusionSolver1D;
	typedef GpuSingleAdvectionDiffusionSolver1D GpuFloatAdvectionDiffusionSolver1D;
	typedef AdvectionDiffusionSolver1D<MemorySpace::Device, MathDomain::Double> GpuDoubleAdvectionDiffusionSolver1D;
	typedef AdvectionDiffusionSolver1D<MemorySpace::Host, MathDomain::Float> CpuSingleAdvectionDiffusionSolver1D;
	typedef CpuSingleAdvectionDiffusionSolver1D CpuFloatAdvectionDiffusionSolver1D;
	typedef AdvectionDiffusionSolver1D<MemorySpace::Host, MathDomain::Double> CpuDoubleAdvectionDiffusionSolver1D;
	typedef GpuSingleAdvectionDiffusionSolver1D ad1D;
	typedef GpuDoubleAdvectionDiffusionSolver1D dad1D;
  
	typedef AdvectionDiffusionSolver2D<MemorySpace::Device, MathDomain::Float> GpuSingleAdvectionDiffusionSolver2D;
	typedef GpuSingleAdvectionDiffusionSolver2D GpuFloatAdvectionDiffusionSolver2D;
	typedef AdvectionDiffusionSolver2D<MemorySpace::Device, MathDomain::Double> GpuDoubleAdvectionDiffusionSolver2D;
	typedef AdvectionDiffusionSolver2D<MemorySpace::Host, MathDomain::Float> CpuSingleAdvectionDiffusionSolver2D;
	typedef CpuSingleAdvectionDiffusionSolver2D CpuFloatAdvectionDiffusionSolver2D;
	typedef AdvectionDiffusionSolver2D<MemorySpace::Host, MathDomain::Double> CpuDoubleAdvectionDiffusionSolver2D;
	typedef GpuSingleAdvectionDiffusionSolver2D ad2D;
	typedef GpuDoubleAdvectionDiffusionSolver2D dad2D;
```

- Wave Equation:
```c++
	typedef WaveEquationSolver1D<MemorySpace::Device, MathDomain::Float> GpuSingleWaveEquationSolver1D;
	typedef GpuSingleWaveEquationSolver1D GpuFloatWaveEquationSolver1D;
	typedef WaveEquationSolver1D<MemorySpace::Device, MathDomain::Double> GpuDoubleWaveEquationSolver1D;
	typedef WaveEquationSolver1D<MemorySpace::Host, MathDomain::Float> CpuSingleWaveEquationSolver1D;
	typedef CpuSingleWaveEquationSolver1D CpuFloatWaveEquationSolver1D;
	typedef WaveEquationSolver1D<MemorySpace::Host, MathDomain::Double> CpuDoubleSolver1D;
	typedef GpuSingleWaveEquationSolver1D wave1D;
	typedef GpuDoubleWaveEquationSolver1D dwave1D;
  
	typedef WaveEquationSolver2D<MemorySpace::Device, MathDomain::Float> GpuSingleWaveEquationSolver2D;
	typedef GpuSingleWaveEquationSolver2D GpuFloatWaveEquationSolver2D;
	typedef WaveEquationSolver2D<MemorySpace::Device, MathDomain::Double> GpuDoubleWaveEquationSolver2D;
	typedef WaveEquationSolver2D<MemorySpace::Host, MathDomain::Float> CpuSingleWaveEquationSolver2D;
	typedef CpuSingleWaveEquationSolver2D CpuFloatWaveEquationSolver2D;
	typedef WaveEquationSolver2D<MemorySpace::Host, MathDomain::Double> CpuDoubleSolver2D;
	typedef GpuSingleWaveEquationSolver2D wave2D;
	typedef GpuDoubleWaveEquationSolver2D dwave2D;
```

## Sample usage - 1D
I wrote a simple python script for plotting the results:

### Hyperbolic - first order: transport equation
- Numerical instability of the centered difference scheme (regardless of the time solver) <p align="center"> <img src="https://raw.githubusercontent.com/pmontalb/PdeFiniteDifferenceSolver/master/instability_compressed.gif"> </p>
- Numerical diffusion induced by the Upwind scheme and solved by the Lax-Wendroff scheme <p align="center"> <img src="https://raw.githubusercontent.com/pmontalb/PdeFiniteDifferenceSolver/master/numericalDiffusion_compressed.gif"> </p>

### Parabolic: heat equation, advection-diffusion equation
- Numerical instability of Explicit Euler scheme <p align="center"> <img src="https://raw.githubusercontent.com/pmontalb/PdeFiniteDifferenceSolver/master/diffusionInstability_compressed.gif"> </p>
- Numerical instability of Gauss-Legendre (4th order diagonally implicit Runge-Kutta) <p align="center"> <img src="https://raw.githubusercontent.com/pmontalb/PdeFiniteDifferenceSolver/master/rungeKuttaFourthOrderDiffusionInstability_compressed.gif"> </p>

### Hyperbolic - second order: wave equation
- Numerical instability of Implicit/Explicit Euler scheme <p align="center"> <img src="https://raw.githubusercontent.com/pmontalb/PdeFiniteDifferenceSolver/master/waveInstability1D_compressed.gif"> </p>

