#pragma once

#include <Vector.h>
#include <IBuffer.h>
#include <Types.h>

#include <PdeInputData.h>

namespace pde
{
	/**
	*	CRTP implementation
	*/
	template<class pdeInputType, MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class FiniteDifferenceSolver
	{
	public:
		FiniteDifferenceSolver(const pdeInputType& inputData)
			: inputData(inputData)
		{

		}

		virtual ~FiniteDifferenceSolver() = default;

	protected:
		pdeInputType inputData;
	};

	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class FiniteDifferenceSolver1D : public FiniteDifferenceSolver<PdeInputData1D<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
		using FiniteDifferenceSolver::FiniteDifferenceSolver;
	};

	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class FiniteDifferenceSolver2D : public FiniteDifferenceSolver<PdeInputData2D<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
		using FiniteDifferenceSolver::FiniteDifferenceSolver;
	};

    #pragma region Type aliases

	typedef FiniteDifferenceSolver1D<MemorySpace::Device, MathDomain::Float> GpuSinglePdeSolver;
	typedef GpuSinglePdeSolver GpuFloatSolver;
	typedef FiniteDifferenceSolver1D<MemorySpace::Device, MathDomain::Double> GpuDoublePdeSolver;

	typedef FiniteDifferenceSolver1D<MemorySpace::Host, MathDomain::Float> CpuSinglePdeSolver;
	typedef CpuSinglePdeSolver CpuFloatSolver;
	typedef FiniteDifferenceSolver1D<MemorySpace::Host, MathDomain::Double> CpuDoubleSolver;

	typedef GpuSinglePdeSolver sol1D;
	typedef GpuDoublePdeSolver dsol1D;

    #pragma endregion
}

