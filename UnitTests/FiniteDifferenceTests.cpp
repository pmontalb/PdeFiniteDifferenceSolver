
#include <gtest/gtest.h>

#include <Vector.h>
#include <ColumnWiseMatrix.h>

#include <FiniteDifferenceSolver.h>

namespace pdet
{
	class FiniteDifferenceTests : public ::testing::Test
	{
	};

	TEST_F(FiniteDifferenceTests, Main)
	{
		cl::vec initialCondition(10, 1.0f);
		cl::vec grid = cl::LinSpace(0.0f, 1.0f, initialCondition.size());
		float velocity = 0.0;
		float diffusion = 1.0;
		double dt = 1e-4;
		SolverType solverType = SolverType::ExplicitEuler;

		pde::GpuSinglePdeInputData data(initialCondition, grid, velocity, diffusion, dt, solverType);
		pde::sol1D solver(data);
	}
}