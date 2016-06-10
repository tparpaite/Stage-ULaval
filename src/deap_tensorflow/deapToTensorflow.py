import tensorflow as tf
from deap import gp

dictArg = dict()

# primitivetree_to_tensor 
#
# Un individu DEAP represente une expression symbolique
# Pour le modeliser on utilise un arbre de type PrimitiveTree
# Cette fonction sert a convertir l'arbre en question vers un
# graphe (un tenseur) interpretable par TensorFlow
#
# Parametres :
# - (PrimitiveTree) individual, l'individu a convertir
# - (int) index, indique l'avancement de la convertion (recursion)
#
# Retour :
# - (Tensor) retVal, le graphe final (lorsque index == 0)
# - (Tensor) retVal, (int) last_id, le graphe intermediaire et l'id du dernier noeud traite

def primitivetree_to_tensor(individual, index = 0):
    x = individual[index]

    if x.arity == 2:
        left_tree, next_id = primitivetree_to_tensor(individual, index + 1)
        right_tree, last_id = primitivetree_to_tensor(individual, next_id + 1)
        retVal = create_tensor_node(x, left_tree, right_tree)

    elif x.arity == 1:
        under_tree, last_id = primitivetree_to_tensor(individual, index + 1)
        retVal = create_tensor_node(x, under_tree, None)

    # x.arity == 0 (leaf)
    else: 
        retVal = create_tensor_node(x, None, None)
        last_id = index

    # root
    if index == 0:
        return retVal
    
    return retVal, last_id


# create_tensor_node
#
# Cette fonction sert a creer un noeud du graphe Tensorflow (tenseur)
# Un noeud peut-etre une operation, auquel cas on indique les parametres
# Un noeud peut aussi etre un argument ou une constante (feuille)
#
#
# Parametres :
# - (Primitive, Terminal, rand101) x
#      | (Primitive) le nom de l'operation
#      | (Terminal) le nom de l'argument
#      | (rand101) la valeur de constante ephemere (coefficient, variable)
# - (Tensor) left_tree, le membre gauche de l'operation
# - (Tensor) right_tree, le membre droit de l'operation
#
# Retour :
# - (Tensor) le noeud correspondant a l'operation ou au terminal

def create_tensor_node(x, left_tree, right_tree):
    # x est un noeud interne (operation)
    if isinstance(x, gp.Primitive):
        op = x.name
        
        if op == "add":
            return tf.add(left_tree, right_tree)
        
        elif op == "sub":
            return tf.sub(left_tree, right_tree)

        elif op == "mul":
            return tf.mul(left_tree, right_tree)

        elif op == "max_rectifier":
            return tf.maximum(left_tree, 0)

        elif op == "min_rectifier":
            return tf.minimum(left_tree, 0)

    # x est une feuille
    else:
        value = x.value
        
        # La feuille correspond a un argument
        if isinstance(x, gp.Terminal):
            # Le noeud qui correspond a l'argument a deja ete cree
            if dictArg.has_key(x.value):
                return dictArg[x.value]
            # Sinon on le cree
            else:               
                node = tf.placeholder("float", [None, 1], name=x.value)
                dictArg[x.value] = node
                return node

        # La feuille correspond a une constante ephemere (var)
        else:
            return tf.Variable(value)



        





