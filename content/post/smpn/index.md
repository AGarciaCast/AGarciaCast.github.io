---
title: "Applications of TopoX to Equivariant Topological Networks"
date: "2024-06-30"
summary: "🔍 Searching for the Best Framework for GDL+TDL Methods? Look no further! This blog post reveals how the TopoX suite boosts modularity and optimizes time and memory usage for methods like Equivariant Simplicial Complexes 🚀"
tags:
  - Demo
  - External
draft: false
image:
  placement: 2

---

<meta http-equiv="refresh" content="0;url=https://gram-blogposts.github.io/2024/blog/2024/smpn/">
<script>window.location.replace("https://gram-blogposts.github.io/2024/blog/2024/smpn/");</script>

Si no se redirige automáticamente, haz clic en este enlace: <a href="https://gram-blogposts.github.io/2024/blog/2024/smpn/">Ir al post externo</a>.





Applications of TopoX to Equivariant Topological Networks
Studying the properties of message passing accross equivariant topological neural networks using the TopoX Suite

Authors
Affiliations
Martin Carrasco

VU Amsterdam

Andreas Berentzen

University of Amsterdam

Alejandro Garcia Castellanos

University of Amsterdam

Published
June 30, 2024

Contents
Introduction
Higher-order networks and why topology is useful
Simplicial Complex: What is it ?
Geometric realization
GNNs and E(n) Equivariant GNNs
Good ol' message passing
E(n) equivariant GNN
Message Passing Simplicial Networks (MPSN)
The new standard: TopoX
1. Building structure
2. Giving meaning to structure
3. The model
4. Application
Experiments
Lifting times
File size
Forward pass
Replication of results
Conclusions
Introduction
Representation learning using Graph Neural Networks (GNNs) is a rapidly growing approach to complex tasks in chemistry 
[1, 2, 3, 4]
. Notably, in a subset of these tasks, a crucial aspect is maintaining equivariance to transformations such as translation, rotation and reflection. Learning representations such that equivariance or invariance can be applied has proved very helpful 
[2, 3]
. Additionally, incorporating higher-order relations in GNNs such that they encode more complex topological spaces is a recent effort to increase the expressivity of GNNs 
[5, 3, 6]
.

This blogpost aims to draw attention to Topological Deep Learning (TDL) by using the suite of Python packages TopoX 
[7]
 to replicate the work of 
[3]
 and show how much simpler development is in this framework. Additionally, we experimented with different topological spaces with more geometric information and compared the results with those of the original work.

Higher-order networks and why topology is useful
Our regular and beloved graphs, as functional as they are, have a bound on the expressive power under message passing networks (MPN) that operate over them 
[8]
. For example, they can learn higher dimensional graph structures, such as cliques. The 1-WL 
[8]
 test is one of the measurements that characterize expressivity in distinguishing non-isomorphic graphs. While this measurement has its own set of limitations, it is the standard in characterizing expressivity.

In search of more expressive structures, the exploration of higher-dimensional topological spaces where higher-dimensional features, such as cliques, can be represented (and learned). Let’s start with what the space of a graph encodes. Let 
 be a graph. We can interpret it as encoding relationships between nodes 
 as a tuple represented by an edge 
. What would relations between pairs of nodes and other pairs 
 or between pairs and triplets 
 would look like and how can we represent them? One space that allows for this kind of relationship is Simplicial Complexes. Extended graphs with features for higher dimensional structures are subjected to constraints. Luckily, the combinatorial definition of these spaces is more straightforward.

Simplicial Complex: What is it ?
An abstract simplicial complex (ASC) is the combinatorial expression of a non-empty set of simplices.

Concretly, let 
 be the powerset of 
 and let 
, then 
 is an ASC if for every 
 and every non-empty 
 it holds that 
. Also, we define the cardinality as  
 
, to be the highest cardinality of a simplex in an ASC minus 1. If the rank is 
, it holds 
.

Click here for an example
Geometric realization
Although an ASC is a purely combinatorial object, it always entails a geometric realization. A Simplicial Complex is the geometric realization of an ASC, constructed out of the underlying geometric points in 
. Thus, Simplicial Complex of dimension 
1
1 is equivalent to a geometric graph (can you see why ?).

The procedure of transforming a structure to a topological domain is commonly called lifting. Thus, lifting can be from point clouds to graphs, graphs to simplicial complexes, or other pairs of domains, given that the properties of the target domain mentioned before hold.

Click here to see the ASC corresponding to Figure 1

Figure 1: Ilustration of the lifting procedure
This shows the lifting procedure of a graph into a simplicial complex using the clique lifting, where nodes are 0-simplices , edges are 1-simplices and triangles are 2-simplices
Clique Complex
Of the plethora of spaces, a relatively simple and intuitive one is the Clique Complex shown at the top left of Figure 1. To describe the Clique Complex we need to formally define a clique, something we skipped over before.

Click here for a formal definition of clique
The Clique Complex is a Simplicial Complex of rank 
. Each clique will become an 
-simplex depending on the cardinality of 
 such that 
. The number of cliques grows exponentially, and the problem of finding all the cliques is complex. For this reason, the naive time complexity of this lift is 

Vietoris-Ripps Complex
The Vietoris-Rips complex is a common way to form a topological space efficiently. The time complexity for generating the procedure depends on the parameter 
 and the number of points 
 given by 
, where 
 is the diameter of the balls grown around points to calculate relationships. This space is equivalent to a Clique Complex in a geometric graph. Below, you can see a visualization of the lifting process with varying values of 
. The points are 0-simplex , the lines are 1-simplex and the triangles are 2-simplex . The growing ball is the disk relating to the 
 parameter.


GNNs and E(n) Equivariant GNNs
GNNs come in many flavors, but to be concise, we will focus on the Message Passing Networks where messages are passed among neighborhoods, which update the node’s representation. At a given number of passes, we will take the node representations and perform classification or regression tasks or pool them to perform the task on the whole graph. Next, we will introduce the remainder of the MPN framework and equivariant GNNs.

Good ol’ message passing
Let 
 be a graph consisting of nodes 
 and edges 
. Then let each node 
 and edge 
 have an associated node feature 
 and edge feature 
, with dimensionality 
. Then, we define a message passing layer as:

 
 
 
  
 

E(n) equivariant GNN
Using inductive biases to steer the training of GNNs towards a particular domain is common. There is a specific class of problems where symmetries are an intrinsic part of their representation. Examples of these are 3D molecular structures and N-body systems. To enhance the learning procedure, we restrict the families of learned functions by guaranteeing equivariance with the action of a transformation from a particular symmetry group. Of of those groups is the 
 group, which encodes rotation, translation, reflection and scaling ( Euclidean group of dimension 
 ).

Tasks: QM9 and N-Body
QM9 is a dataset of 134K small molecules (up to 
 atoms without counting hydrogen) with 
 regression tasks representing the molecule’s properties. Each atom has a set of features and a position in 3D space. Pytorch geometric makes this dataset available in a convenient graph representation. The N-body problem extends the “Charged particle N-body system” to 3D where 
 particles have a positive and negative charge, position, and velocity. Then, the prediction is used to estimate the particle’s position after the steps of 
.

Equivariance and Invariance
Invariance is when an object or set of objects remain the same after a transformation. In contrast, equivariance is a symmetry concerning a function and a transformation. At first glance, these definitions are complicated to picture; however, with some group theory, they will become more apparent.

Let 
 be a group and let 
, 
 be sets on which 
 acts. A function 
 is called equivariant with respect to 
 if it commutes with the group action. Equation 
 expresses this notion formally.

 
Conversly, Equation 
 shows that invariance is when the application of the transformation 
 does not affect the output of the map 
,

 
Do invariances hold in GNNs ?
Equation 
 comes to replace Equation 
 with our invariant function. To make the network equivariant, we introduce feature vector 
, which contains the positional coordinates in Euclidean space. Equation 
 refers to the update in the position embedding of the node. The proof that with this condition, equivariance holds can be found in 
[9]
.  
 

 
 
Message Passing Simplicial Networks (MPSN)
In 
[3]
 authors generalize the 
 equivariant GNN from 
[9]
 to operate on Simplicial Complexes. First, the authors show how relationships between simplices are established via higher-order neighborhoods. Then, they pick a set of equivariant features on the geometric shapes embedded in the topological space up to relations involving 
-simplices (triangles). Finally, they show how equivariance holds under message passing among simplices. We will go over each of these.

Higher-order neighborhoods
We establish some definitions to define proximity relations, such as graph adjacencies in 
-simplex. We will work with only two types of adjacencies as they have proven to be as expressive as using all of them. First, let 
 and 
 be two simplices, we say that 
 is on the bound of 
 or 
 if :

Equation 
 referes to the relation between a 
-simplex and the 
-simplex that compose it. Equation 
 referes to the relationship between 
-simplex and other 
-simplex that are a part of a higher 
-simplex. They are also referred to as cofaces in the literature 
[5]
.

 
 
Equivariant relations
We will make use of previous definitions in the following small section. Recall that we showed that we can choose any 
 such that when applying to a certain class of objects invariant, it is invariant with respect to the group action of any 
 for 
 in this case. The authors of the paper choose four types of message passing. Each of these will have a set of geometric features depending on 
. In general 
 and 
 mean a shared point, 
 and 
 are points not shared.

Num	
1	
2	0	0	
3	0	
4	-	-	
5	-	-	
6	-	-	

Figure 2: Geometric relations
This ilustrations represent how geometric relations are interpreted in terms of boundries (right) and upper-adjacencies (left). In the boundry relationships the figures share all but one point and in the upper-adjacencies they might share some points given that they are part of a higher-order simplex.
Does it work on higher-order networks ?
Using the previous definitions of neighborhoods, 
[3]
 defines a message for each neighborhood as Equation 
 and Equation 
 and replaces the hidden representation update to take these messages into account in Equation 
.

 
 
 
 
 
Finally, they define a graph embedding as Equation 
 where the simplices 
 of each dimension 
 will be aggregated, and the final embedding of the complex will be the concatenation of the embedding of each dimension.

 
 
 
The new standard: TopoX
TopoX is a suite of Python packages that aims to fill the need for accessible and open-source software libraries to handle earning in higher-order domains. In the words of the team behind it, one of its goals is:

facilitate research in topological domains by providing foundational code to understand concepts and offer a platform to disseminate algorithms 
[7]

Given the rapid theoretical advancements in TDL, the need for a solid experimental playground is clear. Many of the development setups would benefit in terms of development efficacy and replication capabilities by sharing a set of standards and practices. TopoX is compromised of four modules: TopoModelX, TopoNetX, TopoEmbbedX and TopoBenchmarkX

TopoModelX: Variety of models based on higher-order message passing built on top of Pytorch
TopoNetX: Similar to NetworkX it provides utilities to handle nodes, edge, higher-order cells and the calculation of adjacencies, incidences and hodge laplacians over complexes
TopoEmbeddX: Embedding of topological domains in euclidean domains
TopoBenchmarkX : Addition of datasets, transform such as lifts and new deep learning models
Next, we illustrate the development process and reproduction of 
[3]
 in the TopoX suite. Additionally, as a base project, we use the ICML TDL Challenge 2024 repo for development, which has a very similar structure to the following package in the suite TopoBenchmarkX 
[10]

1. Building structure
We are first concerned with the lifting of our initial graph or set of points. To perform that task, we will make use of GHUDI 
[11]
, a Python library with many methods mainly used for Topological Data Analysis. We will lift from the graph domain to the simplicial complex domain. Each lifting procedure is a Pytorch BaseTransform with a forward that looks like this.

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        r"""Applies the full lifting (topology + features) to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data
            The lifted data.
        """
        initial_data = data.to_dict()
        lifted_topology = self.lift_topology(data)
        lifted_topology = self.feature_lifting(lifted_topology)
        return torch_geometric.data.Data(**initial_data, **lifted_topology)
In essence, we only need to define a subclass of the source and target domain of our choice (in this case Graph2SimplicialLifting) and override the lift_topology method and the apply feature lifting.

 def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        simplicial_complex = rips_lift(data, self.complex_dim, self.delta)

        feature_dict = {}
        for i, node in enumerate(data.x):
            feature_dict[i] = node

        simplicial_complex.set_simplex_attributes(feature_dict, name='features')

        return self._get_lifted_topology(simplicial_complex, data)
In this case, we introduced rips_lift, which is going to do the actual computation of the lift, and _get_lifted_topology which will transform our SimplicialComplexinto a Pytorch Data object.

def rips_lift(graph: torch_geometric.data.Data, dim: int, dis: float,
                    fc_nodes: bool = True) -> SimplicialComplex:
    x_0, pos = graph.x, graph.pos

    points = [pos[i].tolist() for i in range(pos.shape[0])]

    rips_complex = gudhi.RipsComplex(points=points, max_edge_length=dis)
    simplex_tree: SimplexTree  = rips_complex.create_simplex_tree(max_dimension=dim)

    if fc_nodes:
        nodes = [i for i in range(x_0.shape[0])]
        for edge in combinations(nodes, 2):
            simplex_tree.insert(edge)

    return SimplicialComplex.from_gudhi(simplex_tree)
Note that optionally, nodes are fully connected, per the implementation of 
[3]
. On _lifted_topology, we build the matrix representation of our complex. The library provides the get_complex_connectivityand constructs the connectivity matrices.

    def _get_lifted_topology(
        self, simplicial_complex: SimplicialComplex, graph: nx.Graph
    ) -> dict:
        lifted_topology = get_complex_connectivity(
            simplicial_complex, self.complex_dim, signed=self.signed
        )

        for r in range(0, simplicial_complex.dim+1):
            # Convert to edge_index format
            lifted_topology[f'adjacency_{r}'] = 
            lifted_topology[f'adjacency_{r}'].to_dense().nonzero().t().contiguous()

            lifted_topology[f'incidence_{r}'] = 
            lifted_topology[f'incidence_{r}'].to_dense().nonzero().t().contiguous()


        for r in range(0, simplicial_complex.dim+1):
            # Returns the list of `r`-simplex as `r`-tuples and convert to tensor
            lifted_topology[f'x_idx_{r}'] = 
            torch.tensor(simplicial_complex.skeleton(r), dtype=torch.int)

        lifted_topology["x_0"] = torch.stack(
            list(simplicial_complex.get_simplex_attributes("features", 0).values())
        )

        return lifted_topology

Note that we transform the adjacency and incidence matrices to their edge_index form by using nonzero().t().contigous(). This transformation is to be able to perform mini-batching during training.

2. Giving meaning to structure
Now that we have a higher-order topological space, we are missing only one thing. What should the embeddings of the 
-simplex higher than 
 be ?

There is still the question of what a good feature-lifting technique constitutes and what information it should take from the underlying representation. Authors in 
[3]
 perform an element-wise mean of the components of lower 
-simplex, but the field is open to experimentation. Other alternatives are also being explored. As part of our contribution, we leverage TopoX. This tool allows us to accelerate these calculations by vectorizing this part of the process.

Formally, if 
, then:

Click here for an example
3. The Model
On the model side, the most interesting addition is by TopoModelX and the Conv class. This class represents the convolution operator on the graph, which is necessary for aggregating messages over the neighborhood. We show our implementation of the Conv class to allow equivariant information to pass inside the messages. Everything else is standard Pytorch (unless we use one of the provided models)

    def forward(self, x_source, edge_index, x_weights, x_target=None) -> torch.Tensor:
        # Construct the edge index tensor of size (2, n_boundaries)
        # x_weights is indexed with send_idx because there might be more relationships
        # than r-cells, in that case the weights are not aligned 
        x_message = torch.cat((x_source[send_idx], x_target[recv_idx], x_weights), dim=1) 

        if self.weight_1 is not None:
            x_message = torch.mm(x_message, self.weight_1)
            if self.biases_1 is not None:
                x_message += self.biases_1
            x_message = self.update(x_message)
        if self.weight_2 is not None:
            x_message = torch.mm(x_message, self.weight_2)
            if self.biases_2 is not None:
                x_message += self.biases_2
            x_message = self.update(x_message)
        if self.weight_3 is not None:
            x_message_weights = torch.mm(x_message, self.weight_3)
            if self.biases_3 is not None:
                x_message_weights += self.biases_3
            x_message_weights = torch.nn.functional.sigmoid(x_message_weights)
        else:
            x_message_weights = torch.ones_like(x_message)

        # Weight the message by the learned weights
        x_message = x_message * x_message_weights

        return x_message
4. Application
Now, on to the most important part. We con now very easily execute our model. To load our dataset, it is straightforward

dataset_name = "manual_dataset"
dataset_config = load_dataset_config(dataset_name)
loader = GraphLoader(dataset_config)
dataset = loader.load()
We can also visualize the information in point cloud, simplicial complex, or cell complex domains. Figure 3 is one such of test Simplicial Complex

describe_data(dataset)

After setting the configurations, using a lifting procedure is as easy as defining it’s name.


# Define transformation type and id
transform_type = "liftings"
# If the transform is a topological lifting, 
it should include both the type of the lifting and the identifier
transform_id = "graph2simplicial/empsn_lifting"

# Read yaml file
transform_config = {
    "lifting": load_transform_config(transform_type, transform_id)
    # other transforms (e.g. data manipulations, feature liftings) can be added here
}
lifted_dataset = PreProcessor(dataset, transform_config, loader.data_dir)
Finally, we load our model and execute it.

from modules.models.simplicial.empsn import EMPSNModel

model_type = "simplicial"
model_id = "empsn"
model_config = load_model_config(model_type, model_id)

model = EMPSNModel(model_config, dataset_config)oy_hat = model(lifted_dataset.get(0))
y_hat = model(lifted_dataset.get(0))

Experiments
We performed experiments with the implementation in the TopoX framework. Additionally, we vectorized and improved the following sections: 1) the lifting procedure now results in a smaller file size, and the vectorization of the feature embeddings and adjacency/incidence matrix calculation with the help of TopoX is faster; 2) we organized the computation of the invariants, and further vectorized some operations.

Next, we present a comparison of the lifting times of the whole QM9 dataset, the size of the pre-processed file (after lifting), and the time it takes to run a forward pass.

Lifting times
The lifting procedure has two challenging areas optimized: 1) calculation of the adjacency and incidence relationships for which we rely on TopoX, and 2) feature lifting, which we manually optimized. We brought down the lifting time from about 1.5 hours to about 28 minutes on the same hardware. Next, we show the lifting time of each graph in the dataset and compare the total lifting time. We see that our optimization could be much faster on the heavier graphs. However, there is a set that is reduced to the baseline.



File size
We observed a reduction in the size of the preprocessed dataset after the lifting procedure. This behavior is due to the way incidences and adjacencies are stored. Instead of having the invariance information directly, we can store the relationships, such as boundaries, upper adjacencies, and embeddings, which makes it enough. We could have also stored these as sparse tensors; however, handling the mini-batching proved cumbersome on that representation. Ultimately, we reduced the size from 8.7GB to 6.3 GB.

Forward pass
One of these models’ bottlenecks is the time they take to train for an epoch. The original version took close to 70 hours in the cluster we had access to. Some of this time is related to the number of messages passing and some to calculating the invariances, which must be done each forward pass. We managed to optimize this calculation and thus speed up the execution of the model almost ten-fold. These results are calculated over the forward on a batch size of 
, as per the original implementation, and take into account 
 batches.


Replication of results
On the side of replication, we executed the original, publicly available code shown in Figure 4. Figure 5 compares our implementation and execution using the same hyperparameters set in the code. Due to computing constraints, we could not run experiments on the number of epochs set in the original codebase. For that reason, we report on two particular experiments:

Execution of the original implementation up to epoch ~700.

Figure 4: Base code - Validation MAE
Execution of our implementation up to epoch ~100.

Figure 5: Our code - Validation MAE
Our scores are not near SOTA or the reported scores in the original work. Nevertheless, thanks to the optimizations mentioned above, we were able to execute different tests during the procedure for a reduced number of epochs. Varying batch sizes and learning rates, as well as using other common weight initialization techniques, did not improve the results.

Conclusions
In this post, we superficially introduced the field of topological deep learning and placed it in the field of graph neural networks. Additionally, we investigated the novel development suite for Topological Deep Learning (TopoX) and how it can be used to tackle a particular problem. We review concepts in geometric deep learning and show why they work and how we can leverage topological representations to better learn in message-passing networks. Using the unified TopoX framework allows for ease of development and standardization regarding reproducibility. Additionally, optimization for computationally heavy procedures such as the ones inherent to TDL is more straightforward. Additionally, we replicate the work of 
[3]
. Based on our tests, we could not reach the reported results. However, we took the original repository and replicated the implementation in TopoX. Thus, we cannot know the configurations that achieved the presented results.

References
Attending to Topological Spaces: The Cellular Transformer
Ballester, R., Hernandez-Garcia, P., Papillon, M., Battiloro, C., Miolane, N., Birdal, T., Casacuberta, C., Escalera, S. and Hajij, M., 2024. arXiv preprint arXiv:2405.14094.
Fast, Expressive SE 
(
n
)
(n) Equivariant Networks through Weight-Sharing in Position-Orientation Space
Bekkers, E.J., Vadgama, S., Hesselink, R.D., van der Linden, P.A. and Romero, D.W., 2023. arXiv preprint arXiv:2310.02970.
E (
n
n) Equivariant Message Passing Simplicial Networks
Eijkelboom, F., Hesselink, R. and Bekkers, E.J., 2023. International Conference on Machine Learning, pp. 9071--9081.
E (n) Equivariant Topological Neural Networks
Battiloro, C., Karaismailoglu, E., Tec, M., Dasoulas, G., Audirac, M. and Dominici, F., 2024. arXiv preprint arXiv:2405.15429.
Topological deep learning: Going beyond graph data
Hajij, M., Zamzmi, G., Papamarkou, T., Miolane, N., Guzman-Saenz, A., Ramamurthy, K.N., Birdal, T., Dey, T.K., Mukherjee, S., Samaga, S.N. and others,, 2022. arXiv preprint arXiv:2206.00606.
Topological Neural Networks: Mitigating the Bottlenecks of Graph Neural Networks via Higher-Order Interactions
Giusti, L., 2024. arXiv preprint arXiv:2402.06908.
TopoX: a suite of Python packages for machine learning on topological domains
Hajij, M., Papillon, M., Frantzen, F., Agerberg, J., AlJabea, I., Ballester, R., Battiloro, C., Bernardez, G., Birdal, T., Brent, A. and others,, 2024. arXiv preprint arXiv:2402.02441.
How powerful are graph neural networks?
Xu, K., Hu, W., Leskovec, J. and Jegelka, S., 2018. arXiv preprint arXiv:1810.00826.
E (
n
n) equivariant graph neural networks
Satorras, V.G., Hoogeboom, E. and Welling, M., 2021. International conference on machine learning, pp. 9323--9332.
TopoBenchmarkX: A Framework for Benchmarking Topological Deep Learning
Telyatnikov, L., Bernardez, G., Montagna, M., Vasylenko, P., Zamzmi, G., Hajij, M., Schaub, M.T., Miolane, N., Scardapane, S. and Papamarkou, T., 2024. arXiv preprint arXiv:2406.06642.
GUDHI User and Reference Manual  [link]
Project, T.G., 2021. GUDHI Editorial Board.
© Copyright 2025 .