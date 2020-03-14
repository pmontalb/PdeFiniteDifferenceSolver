#include <forge.h>
#define USE_FORGE_CUDA_COPY_HELPERS
#include <fg/compute_copy.h>

#include <Utils/PdeSetup.h>

static constexpr size_t plotWidth = { 1024 };
static constexpr size_t plotHeight = { 768 };

template<typename solverType, MathDomain md>
void run1D(solverType& solver, const clp::CommandLineArgumentParser& ap, bool debug)
{
	using vType = cl::Vector<MemorySpace::Device, md>;
	using sType = typename vType::stdType;

	// steps to advance before outputing the solution
	auto n = ap.GetArgumentValue<unsigned>("-n");

	// total number of steps
	auto N = ap.GetArgumentValue<unsigned>("-N");

	forge::Window wnd(plotWidth, plotHeight, "Solution");
	wnd.makeCurrent();

	forge::Chart chart(FG_CHART_2D);
	auto& grid = solver.inputData.spaceGrid;
	auto _grid = solver.inputData.spaceGrid.Get();

	static constexpr forge::dtype precision = forge::f32;
	forge::Plot plt = chart.plot(grid.size(), precision, FG_PLOT_LINE, FG_MARKER_NONE);
	plt.setColor(FG_BLUE);

	GfxHandle* handles;
	createGLBuffer(&handles, plt.vertices(), FORGE_VERTEX_BUFFER);

	bool toDo = true;
	bool setLimits = true;
	std::unique_ptr<vType> xyPair = nullptr;
	do
	{
		if (toDo)
		{
			for (unsigned m = 0; m < N; ++m)
			{
				solver.Advance(n);
				if (setLimits)
				{
					setLimits = false;
					auto _initialCondition = solver.solution->columns[0]->Get();
					chart.setAxesLimits(_grid.front(), _grid.back(), *std::min_element(_initialCondition.begin(), _initialCondition.end()), *std::max_element(_initialCondition.begin(), _initialCondition.end()));
				}
				if (!xyPair)
					xyPair = std::make_unique<vType>(2 * grid.size());
				vType::MakePair(*xyPair, grid, *solver.solution->columns[0]);
				copyToGLBuffer(handles, reinterpret_cast<ComputeResourceHandle>(xyPair->GetBuffer().pointer), plt.verticesSize());
				wnd.draw(chart);
			}
		}

		wnd.draw(chart);
		toDo = false;
	}
	while (!wnd.close());
	releaseGLBuffer(handles);
}


template<typename solverType, MathDomain md>
void run2D(solverType& solver, const clp::CommandLineArgumentParser& ap, bool debug)
{
	using vType = cl::Vector<MemorySpace::Device, md>;
	using mType = cl::ColumnWiseMatrix<MemorySpace::Device, md>;
	using sType = typename vType::stdType;

	// steps to advance before outputing the solution
	auto n = ap.GetArgumentValue<unsigned>("-n");

	// total number of steps
	auto N = ap.GetArgumentValue<unsigned>("-N");

	// solution matrix is a collection of flattened solutions over time
	forge::Window wnd(plotWidth, plotHeight, "Solution");
	wnd.makeCurrent();

	forge::Chart chart(FG_CHART_3D);

	auto& xGrid = solver.inputData.xSpaceGrid;
	auto& yGrid = solver.inputData.ySpaceGrid;
	auto _xGrid = xGrid.Get();
	auto _yGrid = yGrid.Get();
	chart.setAxesTitles("x-axis", "y-axis", "Solution");

	forge::Surface surf = chart.surface(_xGrid.size(), _yGrid.size(), forge::f32);
	surf.setColor(FG_BLUE);

	GfxHandle* handle;
	createGLBuffer(&handle, surf.vertices(), FORGE_VERTEX_BUFFER);

	bool toDo = true;
	bool setLimits = true;
	std::unique_ptr<vType> xyzTriple = nullptr;
	do
	{
		if (toDo)
		{
			for (unsigned m = 0; m < N; ++m)
			{
				solver.Advance(n);
				if (setLimits)
				{
					setLimits = false;
					auto _ic = solver.solution->columns[0]->Get();
					chart.setAxesLimits(_xGrid.front(), _xGrid.back(), _yGrid.front(), _yGrid.back(), *std::min_element(_ic.begin(), _ic.end()), *std::max_element(_ic.begin(), _ic.end()));
				}

				if (!xyzTriple)
					xyzTriple =std::make_unique<vType>(3 * xGrid.size() * yGrid.size());

				mType::MakeTriple(*xyzTriple, xGrid, yGrid, *solver.solution->columns[0]);
				copyToGLBuffer(handle, reinterpret_cast<ComputeResourceHandle>(xyzTriple->GetBuffer().pointer), surf.verticesSize());
				wnd.draw(chart);
			}
		}

		wnd.draw(chart);
		toDo = false;
	}
	while (!wnd.close());
	releaseGLBuffer(handle);
}

int main(int argc, char** argv)
{
	clp::CommandLineArgumentParser ap(argc, argv);

	auto mathDomain = ep::ParseMathDomain(ap.GetArgumentValue<std::string>("-md", "Float"));
	auto pdeType = ap.GetArgumentValue<std::string>("-pde", "AdvectionDiffusion");
	auto dimensionality = ap.GetArgumentValue<int>("-dim", 1);
	auto debug = ap.GetFlag("-dbg");

	if (dimensionality == 1)
	{
		switch (mathDomain)
		{
			case MathDomain::Float:
				if (pdeType == "AdvectionDiffusion")
				{
					auto solver = setup1D<pde::AdvectionDiffusionSolver1D<MemorySpace::Device, MathDomain::Float>, MathDomain::Float>(ap, debug);
					run1D<decltype(*solver), MathDomain::Float>(*solver, ap, debug);
				}
				else if (pdeType == "WaveEquation")
				{
					auto solver = setup1D<pde::WaveEquationSolver1D<MemorySpace::Device, MathDomain::Float>, MathDomain::Float>(ap, debug);
					run1D<decltype(*solver), MathDomain::Float>(*solver, ap, debug);
				}
				else
					throw NotImplementedException();
				break;
			case MathDomain::Double:
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
				{
					auto solver = setup2D<pde::AdvectionDiffusionSolver2D<MemorySpace::Device, MathDomain::Float>, MathDomain::Float>(ap, debug);
					run2D<decltype(*solver), MathDomain::Float>(*solver, ap, debug);
				}
				else if (pdeType == "WaveEquation")
				{
					auto solver = setup2D<pde::WaveEquationSolver2D<MemorySpace::Device, MathDomain::Float>, MathDomain::Float>(ap, debug);
					run2D<decltype(*solver), MathDomain::Float>(*solver, ap, debug);
				}
				else
					throw NotImplementedException();
				break;
			case MathDomain::Double:
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