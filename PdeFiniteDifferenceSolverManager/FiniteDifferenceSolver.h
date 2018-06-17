#pragma once

#include <memory>
#include <Vector.h>
#include <IBuffer.h>
#include <Types.h>

#include <FiniteDifferenceManager.h>
#include <CudaException.h>

#define MAKE_DEFAULT_CONSTRUCTORS(CLASS)\
	virtual ~CLASS() noexcept = default;\
	CLASS(const CLASS& rhs) noexcept = default;\
	CLASS(CLASS&& rhs) noexcept = default;\
	CLASS& operator=(const CLASS& rhs) noexcept = default;\
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
		FiniteDifferenceSolver(const pdeInputType& inputData);

		MAKE_DEFAULT_CONSTRUCTORS(FiniteDifferenceSolver);

		void Advance(const unsigned nSteps = 1);

		const cl::Tensor<memorySpace, mathDomain>* const GetTimeDiscretizer() const noexcept;

		std::shared_ptr<cl::ColumnWiseMatrix<memorySpace, mathDomain>> solution;
		const pdeInputType& inputData;
	protected:
		std::shared_ptr<cl::Tensor<memorySpace, mathDomain>> timeDiscretizers;
	};
}

#undef MAKE_DEFAULT_CONSTRUCTORS

#include <FiniteDifferenceSolver.tpp>