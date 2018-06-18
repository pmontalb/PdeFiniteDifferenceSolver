#pragma once

#include <FiniteDifferenceSolver2D.h>

#define MAKE_DEFAULT_CONSTRUCTORS(CLASS)\
	virtual ~CLASS() noexcept = default;\
	CLASS(const CLASS& rhs) noexcept = default;\
	CLASS(CLASS&& rhs) noexcept = default;\
	CLASS& operator=(const CLASS& rhs) noexcept = default;\
	CLASS& operator=(CLASS&& rhs) noexcept = default;

namespace pde
{
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class AdvectionDiffusionSolver2D : public FiniteDifferenceSolver2D<AdvectionDiffusionSolver2D<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		friend class FiniteDifferenceSolver2D<AdvectionDiffusionSolver2D<memorySpace, mathDomain>, memorySpace, mathDomain>;
		using FiniteDifferenceSolver2D::FiniteDifferenceSolver2D;

		MAKE_DEFAULT_CONSTRUCTORS(AdvectionDiffusionSolver2D);

	protected:
		void MakeTimeDiscretizerWorker(const std::shared_ptr<cl::Tensor<memorySpace, mathDomain>>& timeDiscretizers,
									   const cl::ColumnWiseMatrix<memorySpace, mathDomain>& spaceDiscretizer,
									   const FiniteDifferenceInput2D& input)
		{
			pde::detail::MakeTimeDiscretizerAdvectionDiffusion(timeDiscretizers->GetCube(), spaceDiscretizer.GetTile(), input.solverType, input.dt);
		}
	};

#pragma region Type aliases

	typedef AdvectionDiffusionSolver2D<MemorySpace::Device, MathDomain::Float> GpuSingleAdvectionDiffusionSolver2D;
	typedef GpuSingleAdvectionDiffusionSolver2D GpuFloatAdvectionDiffusionSolver2D;
	typedef AdvectionDiffusionSolver2D<MemorySpace::Device, MathDomain::Double> GpuDoubleAdvectionDiffusionSolver2D;
	typedef AdvectionDiffusionSolver2D<MemorySpace::Host, MathDomain::Float> CpuSingleAdvectionDiffusionSolver2D;
	typedef CpuSingleAdvectionDiffusionSolver2D CpuFloatAdvectionDiffusionSolver2D;
	typedef AdvectionDiffusionSolver2D<MemorySpace::Host, MathDomain::Double> CpuDoubleSolver2D;
	typedef GpuSingleAdvectionDiffusionSolver2D sol2D;
	typedef GpuDoubleAdvectionDiffusionSolver2D dsol2D;

#pragma endregion
}

#undef MAKE_DEFAULT_CONSTRUCTORS

#pragma once
