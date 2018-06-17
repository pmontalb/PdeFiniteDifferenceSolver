#pragma once

#include <memory>
#include <Vector.h>
#include <IBuffer.h>
#include <Types.h>

#include <PdeInputData1D.h>
#include <FiniteDifferenceManager.h>
#include <FiniteDifferenceSolver.h>
#include <CudaException.h>

#define MAKE_DEFAULT_CONSTRUCTORS(CLASS)\
	virtual ~CLASS() noexcept = default;\
	CLASS(const CLASS& rhs) noexcept = default;\
	CLASS(CLASS&& rhs) noexcept = default;\
	CLASS& operator=(const CLASS& rhs) noexcept = default;\
	CLASS& operator=(CLASS&& rhs) noexcept = default;

namespace pde
{
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class FiniteDifferenceSolver1D : public FiniteDifferenceSolver<FiniteDifferenceSolver1D<memorySpace, mathDomain>, PdeInputData1D<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		friend class FiniteDifferenceSolver<FiniteDifferenceSolver1D<memorySpace, mathDomain>, PdeInputData1D<memorySpace, mathDomain>, memorySpace, mathDomain>;
		using FiniteDifferenceSolver::FiniteDifferenceSolver;

		MAKE_DEFAULT_CONSTRUCTORS(FiniteDifferenceSolver1D);

	protected:
		void MakeTimeDiscretizer(const std::shared_ptr<cl::Tensor<memorySpace, mathDomain>>& timeDiscretizers, const SolverType solverType);

		void AdvanceImpl(const MemoryTile& solutionTile,
						 const std::shared_ptr<cl::Tensor<memorySpace, mathDomain>>& timeDiscretizers,
						 const SolverType solverType,
						 const unsigned nSteps = 1);

		void Setup(const unsigned solverSteps);
	};

#pragma region Type aliases

	typedef FiniteDifferenceSolver1D<MemorySpace::Device, MathDomain::Float> GpuSinglePdeSolver1D; 
	typedef GpuSinglePdeSolver1D GpuFloatSolver1D; 
	typedef FiniteDifferenceSolver1D<MemorySpace::Device, MathDomain::Double> GpuDoublePdeSolver1D; 
	typedef FiniteDifferenceSolver1D<MemorySpace::Host, MathDomain::Float> CpuSinglePdeSolver1D; 
	typedef CpuSinglePdeSolver1D CpuFloatSolver1D; 
	typedef FiniteDifferenceSolver1D<MemorySpace::Host, MathDomain::Double> CpuDoubleSolver1D; 
	typedef GpuSinglePdeSolver1D sol1D; 
	typedef GpuDoublePdeSolver1D dsol1D;

#pragma endregion
}

#undef MAKE_DEFAULT_CONSTRUCTORS

#include <FiniteDifferenceSolver1D.tpp>