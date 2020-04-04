#pragma once

#include <memory>
#include <Vector.h>
#include <IBuffer.h>
#include <Types.h>

#include <FiniteDifferenceManager.h>
#include <CudaException.h>

#define MAKE_DEFAULT_CONSTRUCTORS(CLASS)\
	virtual ~CLASS() noexcept = default;\
	CLASS(const CLASS& rhs) noexcept = delete;\
	CLASS(CLASS&& rhs) noexcept = default;\
	CLASS& operator=(const CLASS& rhs) noexcept = delete;\
	CLASS& operator=(CLASS&& rhs) noexcept = default;\


namespace pde
{
	/**
	*	CRTP implementation
	*	Instead of using type traits, I decided to pass another template parameter - pdeInputType - for a less verbose code
	*/
	template<class pdeImpl, class pdeInputType, MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class FiniteDifferenceSolver
	{
	public:
		FiniteDifferenceSolver(pdeInputType&& inputData);

		MAKE_DEFAULT_CONSTRUCTORS(FiniteDifferenceSolver)

		void Precompute();
		void Advance(const unsigned nSteps = 1);

		std::unique_ptr<cl::ColumnWiseMatrix<memorySpace, mathDomain>> solution = nullptr;
		pdeInputType inputData;
	protected:
		std::unique_ptr<cl::Tensor<memorySpace, mathDomain>> timeDiscretizers = nullptr;
		std::unique_ptr<cl::ColumnWiseMatrix<memorySpace, mathDomain>> spaceDiscretizer = nullptr;
		std::unique_ptr<cl::ColumnWiseMatrix<memorySpace, mathDomain>> solutionDerivative = nullptr;

		std::unique_ptr<cl::CompressedSparseRowMatrix<memorySpace, mathDomain>> sparseExplicitTimeDiscretizer = nullptr; // only 1-step explicit solvers are supported in sparse format

	private:
		bool hasPrecomputed = false;
	};
}

#undef MAKE_DEFAULT_CONSTRUCTORS

#include <FiniteDifferenceSolver.tpp>

