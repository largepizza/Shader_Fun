#include "App.h"
#include "simulations/GameOfLife.h"
#include "simulations/Particles.h"
#include <iostream>
#include <stdexcept>

int main()
{
    try
    {
        // ── Pick a simulation ─────────────────────────────────────────────
        // Comment/uncomment to switch. Rebuild to apply.
        // App app(std::make_unique<GameOfLife>());
        App app(std::make_unique<Particles>());

        app.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
