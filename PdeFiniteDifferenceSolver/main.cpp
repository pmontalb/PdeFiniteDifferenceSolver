#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <iostream>

#include <Vector.h>
#include <ColumnWiseMatrix.h>

#include <AdvectionDiffusionSolver1D.h>
#include <AdvectionDiffusionSolver2D.h>
#include <WaveEquationSolver1D.h>
#include <WaveEquationSolver2D.h>

#include <Exception.h>
#include "../CommandLineParser.h"
#include "../EnumParser.h"

#ifdef ENABLE_PLOT
	#include <forge.h>
	#define USE_FORGE_CUDA_COPY_HELPERS
	#include <ComputeCopy.h>
#endif

template<class solverImpl, MathDomain md>
void runner1D(const clp::CommandLineArgumentParser& ap, const bool debug, const bool plot)
{
	using vType = cl::Vector<MemorySpace::Device, md>;
	using sType = typename vType::stdType;

	std::chrono::high_resolution_clock::time_point  start, end;

#define DEBUG_PRINT_START(X)\
	if (debug)\
    {\
	    start = std::chrono::high_resolution_clock::now(); \
		std::cout << #X << std::endl;\
	}

#define DEBUG_PRINT_END\
	if (debug)\
	{\
		end = std::chrono::high_resolution_clock::now(); \
		auto elapsedTime = 1e-6 * static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());\
		std::cout << "... Done" << "[ " << elapsedTime << " ms ]" << std::endl;\
	}

#define DEBUG_PRINT(X)\
	if (debug)\
        std::cout << #X << std::endl;

	DEBUG_PRINT_START(Parsing inputs ...)

    #pragma region Parse Inputs

	auto initialConditionFileString = ap.GetArgumentValue<std::string>("-ic", "");
	auto gridFileString = ap.GetArgumentValue<std::string>("-g", "");
	auto outputFileString = ap.GetArgumentValue<std::string>("-of", "sol.cl");

	auto leftBoundaryConditionTypeString = ap.GetArgumentValue<std::string>("-lbct");
	auto leftBoundaryConditionType = ep::ParseBoundaryConditionType(leftBoundaryConditionTypeString);
	auto leftBoundaryConditionValue = ap.GetArgumentValue<sType>("-lbc");

	std::string rightBoundaryConditionTypeString = ap.GetArgumentValue("-rbct", leftBoundaryConditionTypeString);
	auto rightBoundaryConditionType = ep::ParseBoundaryConditionType(rightBoundaryConditionTypeString);
	auto rightBoundaryConditionValue = ap.GetArgumentValue("-rbc", leftBoundaryConditionValue);

	auto solverType = ep::ParseSolverType(ap.GetArgumentValue<std::string>("-st", "CrankNicolson"));
	auto spaceDiscretizerType = ep::ParseSpaceDiscretizer(ap.GetArgumentValue<std::string>("-sdt", "Upwind"));
	if (spaceDiscretizerType == SpaceDiscretizerType::LaxWendroff && solverType != SolverType::ExplicitEuler)
	{
		std::cout << "WARNING: Lax-Wendroff scheme can be applied only with ExplicitEuler -> overriding solver type" << std::endl;
		solverType = SolverType::ExplicitEuler;
	}

	auto diffusion = ap.GetArgumentValue<sType>("-d");
	auto velocity = ap.GetArgumentValue<sType>("-v");
	auto dt = ap.GetArgumentValue<sType>("-dt");

	// steps to advance before outputing the solution
	auto n = ap.GetArgumentValue<unsigned>("-n");

	// total number of steps
	auto N = ap.GetArgumentValue<unsigned>("-N");

#pragma endregion

	DEBUG_PRINT_END

	using vType = cl::Vector<MemorySpace::Device, md>;
	using sType = typename vType::stdType;

	DEBUG_PRINT_START(Creating grid...)

	std::ifstream gridFile(gridFileString);
	vType *grid = nullptr;
	if (!gridFile.is_open())
	{
		DEBUG_PRINT(... creating linspace(-4, 4, 128))
		grid = new vType(vType::LinSpace(sType(-4.0), sType(4.0), 128u));
	}
	else
	{
		DEBUG_PRINT(... reading from file)
		grid = new vType(vType::VectorFromBinaryFile(gridFileString));
	}
	DEBUG_PRINT_END

	DEBUG_PRINT_START(Creating initial condition ...)

	std::ifstream initialConditionFile(initialConditionFileString);
	vType *initialCondition = nullptr;
	if (!initialConditionFile.is_open())
	{
		DEBUG_PRINT(... creating bell function)
		auto _grid = grid->Get();
		std::vector<sType> bellFunction(grid->size());
		for (unsigned i = 0; i < bellFunction.size(); ++i)
			bellFunction[i] = std::exp(static_cast<sType>(-.25) * _grid[i] * _grid[i]);

		initialCondition = new vType(bellFunction);
	}
	else
	{
		DEBUG_PRINT(... reading from file)
		initialCondition = new vType(vType::VectorFromBinaryFile(initialConditionFileString));
	}
	DEBUG_PRINT_END

	BoundaryCondition leftBc(leftBoundaryConditionType, static_cast<double>(leftBoundaryConditionValue));
	BoundaryCondition rightBc(rightBoundaryConditionType, static_cast<double>(rightBoundaryConditionValue));
	BoundaryCondition1D bc(leftBc, rightBc);

	DEBUG_PRINT_START(Creating PDE input data ...)
	pde::PdeInputData1D<MemorySpace::Device, md> data(*initialCondition, *grid, velocity, diffusion, dt, solverType, spaceDiscretizerType, bc);
	DEBUG_PRINT_END

	DEBUG_PRINT_START(Creating PDE solver ...)
	solverImpl solver(data);
	DEBUG_PRINT_END

	std::vector<sType> solutionMatrix;

	if (!plot)
	{
		unsigned nSolutions = 0;
		DEBUG_PRINT_START(Solving ...)

		for (unsigned m = 0; m < N; ++m)
		{
			solver.Advance(n);

			const auto solution = solver.solution->columns[0]->Get();
			solutionMatrix.insert(solutionMatrix.end(), solution.begin(), solution.end());
			++nSolutions;
		}

		DEBUG_PRINT_END

		DEBUG_PRINT_START(Saving to file ...)
		std::ofstream outputFile(outputFileString);
		cl::MatrixToBinaryFile<sType>(solutionMatrix, nSolutions, initialCondition->size(), outputFileString, false);
		DEBUG_PRINT_END
	}
	else
	{
		#ifdef ENABLE_PLOT
			forge::Window wnd(1000, 800, "Plotting Demo");
			wnd.makeCurrent();

			forge::Chart chart(FG_CHART_2D);
			auto _grid = grid->Get();
			auto _initialCondition = initialCondition->Get();
			chart.setAxesLimits(_grid.front(), _grid.back(), *std::min_element(_initialCondition.begin(), _initialCondition.end()), *std::max_element(_initialCondition.begin(), _initialCondition.end()));

			static constexpr forge::dtype precision = forge::f32;
			forge::Plot plt = chart.plot(grid->size(), precision, FG_PLOT_LINE, FG_MARKER_NONE);
			plt.setColor(FG_BLUE);

			GfxHandle* handles;
			createGLBuffer(&handles, plt.vertices(), FORGE_VERTEX_BUFFER);

			bool toDo = true;
			cl::Vector<MemorySpace::Device, MathDomain::Float>* xyPair = nullptr;
			do
			{
				if (toDo)
				{
					for (unsigned m = 0; m < N; ++m)
					{
						solver.Advance(n);

						if (!xyPair)
							xyPair = new cl::Vector<MemorySpace::Device, MathDomain::Float>(2 * grid->size());
						cl::MakePair(*xyPair, *grid, *solver.solution->columns[0]);
						copyToGLBuffer(handles, static_cast<ComputeResourceHandle>(xyPair->GetBuffer().pointer), plt.verticesSize());
						wnd.draw(chart);
					}
				}

				wnd.draw(chart);
				toDo = false;
			}
			while (!wnd.close());
			releaseGLBuffer(handles);
		#endif
	}

	delete initialCondition;
	delete grid;

#undef DEBUB_PRINT
}

template<class solverImpl, MathDomain md>
void runner2D(const clp::CommandLineArgumentParser& ap, const bool debug, const bool plot)
{
	using vType = cl::Vector<MemorySpace::Device, md>;
	using mType = cl::ColumnWiseMatrix<MemorySpace::Device, md>;
	using sType = typename vType::stdType;

	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

	DEBUG_PRINT_START(Parsing inputs ...)

    #pragma region Parse Inputs

	auto initialConditionFileString = ap.GetArgumentValue<std::string>("-ic", "");
	auto xGridFileString = ap.GetArgumentValue<std::string>("-gx", "");
	auto yGridFileString = ap.GetArgumentValue<std::string>("-gy", "");
	auto outputFileString = ap.GetArgumentValue<std::string>("-of", "sol.cl");

    #pragma region BC

	auto leftBoundaryConditionTypeString = ap.GetArgumentValue<std::string>("-lbct");
	auto leftBoundaryConditionType = ep::ParseBoundaryConditionType(leftBoundaryConditionTypeString);
	sType leftBoundaryConditionValue = ap.GetArgumentValue<sType>("-lbc");

	std::string rightBoundaryConditionTypeString = ap.GetArgumentValue("-rbct", leftBoundaryConditionTypeString);
	auto rightBoundaryConditionType = ep::ParseBoundaryConditionType(rightBoundaryConditionTypeString);
	sType rightBoundaryConditionValue = ap.GetArgumentValue("-rbc", leftBoundaryConditionValue);

	auto downBoundaryConditionTypeString = ap.GetArgumentValue<std::string>("-dbct", leftBoundaryConditionTypeString);
	auto downBoundaryConditionType = ep::ParseBoundaryConditionType(downBoundaryConditionTypeString);
	sType downBoundaryConditionValue = ap.GetArgumentValue<sType>("-dbc", leftBoundaryConditionValue);

	auto upBoundaryConditionTypeString = ap.GetArgumentValue<std::string>("-ubct", leftBoundaryConditionTypeString);
	auto upBoundaryConditionType = ep::ParseBoundaryConditionType(upBoundaryConditionTypeString);
	sType upBoundaryConditionValue = ap.GetArgumentValue<sType>("-ubc", leftBoundaryConditionValue);

    #pragma endregion

	auto solverType = ep::ParseSolverType(ap.GetArgumentValue<std::string>("-st", "CrankNicolson"));
	auto spaceDiscretizerType = ep::ParseSpaceDiscretizer(ap.GetArgumentValue<std::string>("-sdt", "Upwind"));
	if (spaceDiscretizerType == SpaceDiscretizerType::LaxWendroff && solverType != SolverType::ExplicitEuler)
	{
		std::cout << "WARNING: Lax-Wendroff scheme can be applied only with ExplicitEuler -> overriding solver type" << std::endl;
		solverType = SolverType::ExplicitEuler;
	}

	auto diffusion = ap.GetArgumentValue<sType>("-d");
	auto xVelocity = ap.GetArgumentValue<sType>("-vx");
	auto yVelocity = ap.GetArgumentValue<sType>("-vy");
	auto dt = ap.GetArgumentValue<sType>("-dt");

	// steps to advance before outputing the solution
	auto n = ap.GetArgumentValue<unsigned>("-n");

	// total number of steps
	auto N = ap.GetArgumentValue<unsigned>("-N");

    #pragma endregion

	DEBUG_PRINT_END

    #pragma region Grid

	DEBUG_PRINT_START(Creating x grid...)

	std::ifstream xGridFile(xGridFileString);
	vType *xGrid = nullptr;
	constexpr unsigned defaultSize = 10u;
	if (!xGridFile.is_open())
	{
		DEBUG_PRINT(... creating linspace(-4, 4, 128))
		xGrid = new vType(vType::LinSpace(sType(-4.0), sType(4.0), defaultSize));
	}
	else
	{
		DEBUG_PRINT(... reading from file)
		xGrid = new vType(vType::VectorFromBinaryFile(xGridFileString));
	}
	DEBUG_PRINT_END

	DEBUG_PRINT_START(Creating y grid...)

	std::ifstream yGridFile(yGridFileString);
	vType *yGrid = nullptr;
	if (!yGridFile.is_open())
	{
		DEBUG_PRINT(... creating linspace(-4, 4, 128))
		yGrid = new vType(vType::LinSpace(sType(-4.0), sType(4.0), defaultSize));
	}
	else
	{
		DEBUG_PRINT(... reading from file)
		yGrid = new vType(vType::VectorFromBinaryFile(yGridFileString));
	}
	DEBUG_PRINT_END

    #pragma endregion

	DEBUG_PRINT_START(Creating initial condition ...)

	std::ifstream initialConditionFile(initialConditionFileString);
	mType *initialCondition = nullptr;
	if (!initialConditionFile.is_open())
	{
		DEBUG_PRINT(... creating bell function)
		auto _xGrid = xGrid->Get();
		auto _yGrid = yGrid->Get();
		std::vector<sType> bellFunction(xGrid->size() * yGrid->size());
		for (unsigned j = 0; j < _yGrid.size(); ++j)
			for (unsigned i = 0; i < _xGrid.size(); ++i)
				bellFunction[i + _xGrid.size() * j] = std::exp(static_cast<sType>(-.25) * (_xGrid[i] * _xGrid[i] + _yGrid[j] * _yGrid[j]));

		initialCondition = new mType(bellFunction, xGrid->size(), yGrid->size());
	}
	else
	{
		DEBUG_PRINT(... reading from file)
		initialCondition = new mType(mType::MatrixFromBinaryFile(initialConditionFileString));
	}
	DEBUG_PRINT_END

	BoundaryCondition leftBc(leftBoundaryConditionType, static_cast<double>(leftBoundaryConditionValue));
	BoundaryCondition rightBc(rightBoundaryConditionType, static_cast<double>(rightBoundaryConditionValue));
	BoundaryCondition downBc(downBoundaryConditionType, static_cast<double>(downBoundaryConditionValue));
	BoundaryCondition upBc(upBoundaryConditionType, static_cast<double>(upBoundaryConditionValue));
	BoundaryCondition2D bc(leftBc, rightBc, downBc, upBc);

	DEBUG_PRINT_START(Creating PDE input data ...)
	pde::PdeInputData2D<MemorySpace::Device, md> data(*initialCondition, *xGrid, *yGrid, 
													  xVelocity, yVelocity, diffusion, dt, solverType, spaceDiscretizerType, bc);
	DEBUG_PRINT_END

	DEBUG_PRINT_START(Creating PDE solver ...)
	solverImpl solver(data);
	DEBUG_PRINT_END

	// solution matrix is a collection of flattened solutions over time
	std::vector<sType> solutionMatrix;

	if (!plot)
	{
		unsigned nSolutions = 0;
		for (unsigned m = 0; m < N; ++m)
		{
			DEBUG_PRINT_START(Solving ...)
			solver.Advance(n);
			DEBUG_PRINT_END

			const auto solution = solver.solution->columns[0]->Get();
			solutionMatrix.insert(solutionMatrix.end(), solution.begin(), solution.end());
			++nSolutions;
		}

		DEBUG_PRINT_START(Saving to file ...)
		std::ofstream outputFile(outputFileString);
		cl::MatrixToBinaryFile<sType>(solutionMatrix, nSolutions, initialCondition->size(), outputFileString, true);
		DEBUG_PRINT_END
	}
	else
	{
		#ifdef ENABLE_PLOT
			static constexpr size_t plotWidth = { 1024 };
			static constexpr size_t plotHeight = { 768 };
			// solution matrix is a collection of flattened solutions over time
			forge::Window wnd(1024, 768, "Solution");
			wnd.makeCurrent();

			forge::Chart chart(FG_CHART_3D);

			auto _xGrid = xGrid->Get();
			auto _yGrid = yGrid->Get();
			auto _ic = initialCondition->Get();
			chart.setAxesLimits(_xGrid.front(), _xGrid.back(), _yGrid.front(), _yGrid.back(), *std::min_element(_ic.begin(), _ic.end()), *std::max_element(_ic.begin(), _ic.end()));
			chart.setAxesTitles("x-axis", "y-axis", "Solution");

			forge::Surface surf = chart.surface(_xGrid.size(), _yGrid.size(), forge::f32);
			surf.setColor(FG_BLUE);

			GfxHandle* handle;
			createGLBuffer(&handle, surf.vertices(), FORGE_VERTEX_BUFFER);

			bool toDo = true;
			cl::Vector<MemorySpace::Device, MathDomain::Float>* xyzTriple = nullptr;
			do
			{
				if (toDo)
				{
					for (unsigned m = 0; m < N; ++m)
					{
						solver.Advance(n);

						if (!xyzTriple)
							xyzTriple = new cl::Vector<MemorySpace::Device, MathDomain::Float>(3 * xGrid->size() * yGrid->size());

						cl::MakeTriple(*xyzTriple, *xGrid, *yGrid, *solver.solution->columns[0]);
						copyToGLBuffer(handle, (ComputeResourceHandle)xyzTriple->GetBuffer().pointer, surf.verticesSize());
						wnd.draw(chart);
					}
				}

				wnd.draw(chart);
				toDo = false;
			}
			while (!wnd.close());
			releaseGLBuffer(handle);
		#endif
	}

	delete initialCondition;
	delete xGrid;
	delete yGrid;

#undef DEBUB_PRINT
}


int main(int argc, char** argv)
{
	clp::CommandLineArgumentParser ap(argc, argv);

	auto mathDomain = ep::ParseMathDomain(ap.GetArgumentValue<std::string>("-md", "Float"));
	auto pdeType = ap.GetArgumentValue<std::string>("-pde", "AdvectionDiffusion");
	auto dimensionality = ap.GetArgumentValue<int>("-dim", 1);
	auto debug = ap.GetFlag("-dbg");
	auto plot = ap.GetFlag("-plt");

	if (dimensionality == 1)
	{
		switch (mathDomain)
		{
			case MathDomain::Float:
				if (pdeType == "AdvectionDiffusion")
					runner1D<pde::AdvectionDiffusionSolver1D<MemorySpace::Device, MathDomain::Float>, MathDomain::Float>(ap, debug, plot);
				else if (pdeType == "WaveEquation")
					runner1D<pde::WaveEquationSolver1D<MemorySpace::Device, MathDomain::Float>, MathDomain::Float>(ap, debug, plot);
				else
					throw NotImplementedException();
				break;
			case MathDomain::Double:
				if (pdeType == "AdvectionDiffusion")
					runner1D<pde::AdvectionDiffusionSolver1D<MemorySpace::Device, MathDomain::Double>, MathDomain::Double>(ap, debug, plot);
				else if (pdeType == "WaveEquation")
					runner1D<pde::WaveEquationSolver1D<MemorySpace::Device, MathDomain::Double>, MathDomain::Double>(ap, debug, plot);
				else
					throw NotImplementedException();
				break;
			default:
				throw NotImplementedException();
		}
	}
	else if (dimensionality == 2)
	{
		switch (mathDomain)
		{
			case MathDomain::Float:
				if (pdeType == "AdvectionDiffusion")
					runner2D<pde::AdvectionDiffusionSolver2D<MemorySpace::Device, MathDomain::Float>, MathDomain::Float>(ap, debug, plot);
				else if (pdeType == "WaveEquation")
					runner2D<pde::WaveEquationSolver2D<MemorySpace::Device, MathDomain::Float>, MathDomain::Float>(ap, debug, plot);
				else
					throw NotImplementedException();
				break;
			case MathDomain::Double:
				if (pdeType == "AdvectionDiffusion")
					runner2D<pde::AdvectionDiffusionSolver2D<MemorySpace::Device, MathDomain::Double>, MathDomain::Double>(ap, debug, plot);
				else if (pdeType == "WaveEquation")
					runner2D<pde::WaveEquationSolver2D<MemorySpace::Device, MathDomain::Double>, MathDomain::Double>(ap, debug, plot);
				else
					throw NotImplementedException();
				break;
			default:
				throw NotImplementedException();
		}
	}
	else
	{
		throw NotImplementedException();
	}

	return 0;
}
