from deap import gp

# compile_with_weights
#
# Compile l'expression symbolique expr en ajoutant l'impact 
# des poids sur les differents parametres 
# La fonction retournee prend en dernier argument un tableau
# contenant la valeur des poids (coefficients)
#
# Parametres :
# - (PrimitiveTree) exp, l'expression a compiler
# - (PrimitiveSet) pset, l'ensemble des primitives necessaires a la compilation
#
# Retour :
# - (bin) eval(...), une fonction prenant en dernier parametre la liste des poids
#                    si PrimitiveSet a au moins un argument,
#                    sinon on retourne l'evaluation de l'arbre
# - (int) w_size, le nombre de poids a appliquer


def compile_with_weights(expr, pset):
    code, w_size = exp_to_string_with_weights(expr)

    if len(pset.arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        args = ",".join(arg for arg in pset.arguments)
        args += ",w"
        code = "lambda {args}: {code}".format(args=args, code=code)

    try:
        return eval(code, pset.context, {}), w_size
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError, ("DEAP : Error in tree evaluation :"
                            " Python cannot evaluate a tree higher than 90. "
                            "To avoid this problem, you should use bloat control on your "
                            "operators. See the DEAP documentation for more information. "
                            "DEAP will now abort."), traceback


# exp_to_string_with_weights 
#
# Un individu DEAP represente une expression symbolique
# Pour le modeliser on utilise un arbre de type PrimitiveTree
# Cette fonction sert a convertir l'arbre en question vers un
# une chaine de caracteres evaluable en Python comme code executable
# De plus, on prend soin de multiplier chaque argument (terminal)
# par un coefficient (son poids, de base il est de 1)
#
#
# Parametres :
# - (PrimitiveTree) individual, l'individu a convertir
# - (int) index, indique l'avancement de la convertion (recursion)
# - (int) w_index, indique l'indice du poids courant dans le tableau des coefficients
#
# Retour :
# - (str) retVal, (int) w_index la chaine finale (lorsque index == 0) et le nombre de coef
# - (str) retVal, (int) last_id, (int) w_index 
#   La chaine intermediaire, l'id du dernier noeud traite et l'id du dernier poids traite

def exp_to_string_with_weights(individual, index = 0, w_index = 0):
    x = individual[index]

    if x.arity == 2:
        left_tree, next_id, w_index = exp_to_string_with_weights(individual, index + 1, w_index)
        right_tree, last_id, w_index = exp_to_string_with_weights(individual, next_id + 1, w_index)
        retVal = create_string(x, left_tree, right_tree, w_index)

    elif x.arity == 1:
        under_tree, last_id, w_index = exp_to_string_with_weights(individual, index + 1, w_index)
        retVal = create_string(x, under_tree, None, w_index)

    # x.arity == 0 (leaf)
    else: 
        retVal = create_string(x, None, None, w_index)
        w_index += 1
        last_id = index

    # root
    if index == 0:
        return retVal, w_index
    
    return retVal, last_id, w_index


# create_string
#
# Cette fonction sert a creer une chaine de caractere (str)
# Une chaine peut correspondre a une operation
# Une chaine peut aussi correspondre a un argument, 
# et dans ce cas on le multiplie par un coefficient
#
#
# Parametres :
# - (Primitive, Terminal, rand101) x
#      | (Primitive) le nom de l'operation
#      | (Terminal) le nom de l'argument
#      | (rand101) la valeur de constante ephemere (coefficient, variable)
# - (str) left_tree, la chaine correspondant au membre gauche de l'operation
# - (str) right_tree, la chaine correspondant au membre droit de l'operation
# - (int) w_index, l'indice du poids courant dans le tableau des coefficients
#
# Retour :
# - (str) la chaine correspondant a l'operation ou au terminal

def create_string(x, left_tree, right_tree, w_index):
    # x est un noeud interne (operation)
    if isinstance(x, gp.Primitive):
        op = x.name
        
        if op == "add":
            return "add({}, {})".format(left_tree, right_tree)
        
        elif op == "sub":
            return "sub({}, {})".format(left_tree, right_tree)

        elif op == "mul":
            return "mul({}, {})".format(left_tree, right_tree)

        elif op == "max_rectifier":
            return "max_rectifier({})".format(left_tree)

        elif op == "min_rectifier":
            return "min_rectifier({})".format(left_tree)

    # x est une feuille
    else:
        value = x.value
        
        # La feuille correspond a un argument
        if isinstance(x, gp.Terminal):
            return "w[{}]*{}".format(w_index, value)

        # La feuille correspond a une constante ephemere (var)
        else:
            return "w[{}]*{}".format(w_index, value)
