
#include <gtest/gtest.h>

#include <Vector.h>
#include <ColumnWiseMatrix.h>

#include <FiniteDifferenceSolver.h>

namespace pdet
{
	class FiniteDifferenceTests : public ::testing::Test
	{
	};

	TEST_F(FiniteDifferenceTests, ConstantSolutionNoTransportNoDiffusion)
	{
		cl::vec initialCondition(10, 1.0f);
		cl::vec grid = cl::LinSpace(0.0f, 1.0f, initialCondition.size());
		float velocity = 0.0;
		float diffusion = 0.0;
		double dt = 1e-4;
		SolverType solverType = SolverType::ExplicitEuler;

		pde::GpuSinglePdeInputData data(initialCondition, grid, velocity, diffusion, dt, solverType);
		pde::sol1D solver(data);

		const auto _initialCondition = initialCondition.Get();
		for (unsigned n = 0; n < 10; ++n)
		{
			solver.Advance();
			const auto solution = solver.solution->columns[0]->Get();

			for (size_t i = 0; i < solution.size(); ++i)
				ASSERT_TRUE(fabs(solution[i] - _initialCondition[i]) <= 1e-7);
		}
	}
}