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
	/**
	* CRTP implementation
	*/
	template<class solverImpl, MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class FiniteDifferenceSolver1D : public FiniteDifferenceSolver<solverImpl, PdeInputData1D<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		friend class FiniteDifferenceSolver<solverImpl, PdeInputData1D<memorySpace, mathDomain>, memorySpace, mathDomain>;
		using FiniteDifferenceSolver<solverImpl, PdeInputData1D<memorySpace, mathDomain>, memorySpace, mathDomain>::FiniteDifferenceSolver;

		MAKE_DEFAULT_CONSTRUCTORS(FiniteDifferenceSolver1D)

	protected:
		void AdvanceImpl(cl::ColumnWiseMatrix<memorySpace, mathDomain>& solution_,
						 const cl::Tensor<memorySpace, mathDomain>& timeDiscretizers_,
						 const SolverType solverType,
						 const unsigned nSteps = 1);
		void AdvanceImpl(cl::ColumnWiseMatrix<memorySpace, mathDomain>& solution_,
		                 cl::CompressedSparseRowMatrix<memorySpace, mathDomain>& timeDiscretizer_,
		                 const SolverType solverType,
		                 const unsigned nSteps = 1);

		virtual void Setup(const unsigned solverSteps);
	};
}

#undef MAKE_DEFAULT_CONSTRUCTORS

#include <FiniteDifferenceSolver1D.tpp>
