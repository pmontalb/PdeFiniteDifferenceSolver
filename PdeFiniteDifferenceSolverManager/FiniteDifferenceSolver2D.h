#pragma once

#include <FiniteDifferenceSolver.h>

#include <memory>
#include <Vector.h>
#include <IBuffer.h>
#include <Types.h>

#include <PdeInputData2D.h>
#include <FiniteDifferenceManager.h>
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
	class FiniteDifferenceSolver2D : public FiniteDifferenceSolver<FiniteDifferenceSolver2D<memorySpace, mathDomain>, PdeInputData2D<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		friend class FiniteDifferenceSolver<FiniteDifferenceSolver2D<memorySpace, mathDomain>, PdeInputData2D<memorySpace, mathDomain>, memorySpace, mathDomain>;
		using FiniteDifferenceSolver::FiniteDifferenceSolver;

		MAKE_DEFAULT_CONSTRUCTORS(FiniteDifferenceSolver2D);

	protected:
		void MakeTimeDiscretizer(const std::shared_ptr<cl::Tensor<memorySpace, mathDomain>>& timeDiscretizers, const SolverType solverType);

		void AdvanceImpl(const MemoryTile& solutionTile,
						 const std::shared_ptr<cl::Tensor<memorySpace, mathDomain>>& timeDiscretizers,
						 const SolverType solverType,
						 const unsigned nSteps = 1);

		void Setup(const unsigned solverSteps);
	};

#pragma region Type aliases

	typedef FiniteDifferenceSolver2D<MemorySpace::Device, MathDomain::Float> GpuSinglePdeSolver2D; 
	typedef GpuSinglePdeSolver2D GpuFloatSolver2D; 
	typedef FiniteDifferenceSolver2D<MemorySpace::Device, MathDomain::Double> GpuDoublePdeSolver2D; 
	typedef FiniteDifferenceSolver2D<MemorySpace::Host, MathDomain::Float> CpuSinglePdeSolver2D; 
	typedef CpuSinglePdeSolver2D CpuFloatSolver2D; 
	typedef FiniteDifferenceSolver2D<MemorySpace::Host, MathDomain::Double> CpuDoubleSolver2D; 
	typedef GpuSinglePdeSolver2D sol2D; 
	typedef GpuDoublePdeSolver2D dsol2D;

#pragma endregion
}

#undef MAKE_DEFAULT_CONSTRUCTORS

#include <FiniteDifferenceSolver2D.tpp>
