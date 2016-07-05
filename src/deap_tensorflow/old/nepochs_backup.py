# On optimise plus en profondeur pour les 10% meilleurs individus
# top_individuals = tools.selBest(offspring, len(population)/10)
# fitnesses = toolbox.map(toolbox.evaluate, top_individuals, [n_epochs_opti] * len(top_individuals))
# for ind, fit in zip(top_individuals, fitnesses):
#    ind.fitness.values = fit


 # Nouvel individu : on genere des coefficients aleatoires
    if not(individual.fitness.valid):
        individual.optimized_weights = [random.normalvariate(0, 1) for _ in range (n_weights)]

    # Optimisation des constantes avec TensorFlow sur l'ensemble train
    individual_tensor = gpdt.primitivetree_to_tensor(individual, n_args, n_weights, individual.optimized_weights)
    optimized_weights = gpdt.tensorflow_run(individual_tensor, trX, trY, teX, teY, n_epochs)

    # On garde seulement l'optimisation si la fitness est reduite
    if individual.fitness.valid and mean_squarred_error(func, optimized_weights, teX, teY)[0] < individual.fitness.values[0]:
        individual.optimized_weights = optimized_weights
