// Ahmad F. Al Musawi, VCU, 2023

#include <iostream>
#include <fstream>
#include <sstream>

#include <forward_list>
#include <algorithm>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <set>
#include <list>
#include <cmath>
#include <chrono> 
#include <utility>

double CalculateAUPRC(std::vector<std::pair<float, int>> &data)
{
    // Sort data by prediction score in descending order
    std::sort(data.begin(), data.end(), [](const auto &a, const auto &b)
              { return a.first > b.first; });

    size_t tp = 0, fp = 0, total_positives = 0;
    for (const auto &[score, label] : data)
    {
        if (label == 1)
        {
            ++total_positives;
        }
    }

    double auprc = 0.0, prev_recall = 0.0, prev_precision = 1.0;
    for (const auto &[score, label] : data)
    {
        if (label == 1)
        {
            ++tp;
        }
        else
        {
            ++fp;
        }

        double recall = static_cast<double>(tp) / total_positives;
        double precision = static_cast<double>(tp) / (tp + fp);

        // Calculate area under the curve using trapezoidal rule
        auprc += (recall - prev_recall) * (precision + prev_precision) / 2.0;

        prev_recall = recall;
        prev_precision = precision;
    }

    return auprc;
}

double CalculateAUC(std::vector<std::pair<float, int>> &data)
{
    // Sort data by prediction score in descending order
    std::sort(data.begin(), data.end(), [](const auto &a, const auto &b)
              { return a.first > b.first; });

    size_t tp = 0, fp = 0;
    size_t positive = 0, negative = 0;
    for (const auto &[score, label] : data)
    {
        if (label == 1)
            ++positive;
        else
            ++negative;
    }

    double auc = 0.0, prev_fpr = 0.0, prev_tpr = 0.0;
    for (const auto &[score, label] : data)
    {
        if (label == 1)
            ++tp;
        else
            ++fp;

        double fpr = static_cast<double>(fp) / negative;
        double tpr = static_cast<double>(tp) / positive;

        // Add area of trapezoid
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;

        prev_fpr = fpr;
        prev_tpr = tpr;
    }

    return auc;
}

void shuffleVector(std::vector<std::string> &vec)
{
    // Obtain a time-based seed:
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // Shuffle using std::shuffle
    std::shuffle(vec.begin(), vec.end(), std::default_random_engine(seed));
}

int getIndex(std::vector<int> Nodes, int node)
{
    int temp = -1;
    for (size_t i = 0; i < Nodes.size(); ++i)
    {
        if (Nodes[i] == node)
        {
            temp = i;
            break;
        }
    }
    return temp;
}

__global__ void kernel_AA(bool *A, int *neighbors, int *offsets, float *results, int *Nodes, int numNodes)
{
    // implementing the metric
    // Adamic Adar =  \sum_{z \in N_u \cap N_v} \frac{1}{log K_z}

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // indexes of the neighbors of the first node
    int start1 = offsets[col];
    int end1 = offsets[col + 1];

    // indexes of the neighbors of the second node
    int start2 = offsets[row];
    int end2 = offsets[row + 1];

    float total = 0;
    if (row < numNodes && col < numNodes)
    {
        // printf("nodes (%d, %d)\n", row, col);
        if (not A[row * numNodes + col])
        {
            for (int i = start1; i < end1; ++i)
            {
                for (int j = start2; j < end2; ++j)
                {
                    if (neighbors[i] == neighbors[j])
                    {
                        int Kz = 0;
                        for (int t = 0; t < numNodes; t++)
                        {
                            if (Nodes[t] == neighbors[i])
                            {
                                Kz = offsets[t + 1] - offsets[t];
                                break;
                            }
                        }
                        total = total + (1 / log10(Kz));
                        break;
                    }
                }
            }
            results[row * numNodes + col] = total;
        }
    }
}
__global__ void kernel_RA(bool *A, int *neighbors, int *offsets, float *results, int *Nodes, int numNodes)
{
    // implementing the metric
    // Resource Allocation =  \sum_{z \in N_u \cap N_v} \frac{1}{K_z}

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // indexes of the neighbors of the first node
    int start1 = offsets[col];
    int end1 = offsets[col + 1];

    // indexes of the neighbors of the second node
    int start2 = offsets[row];
    int end2 = offsets[row + 1];

    float total = 0;
    if (row < numNodes && col < numNodes)
    {
        // printf("nodes (%d, %d)\n", row, col);
        if (not A[row * numNodes + col])
        {
            for (int i = start1; i < end1; ++i)
            {
                for (int j = start2; j < end2; ++j)
                {
                    if (neighbors[i] == neighbors[j])
                    {
                        int Kz = 0;
                        for (int t = 0; t < numNodes; t++)
                        {
                            if (Nodes[t] == neighbors[i])
                            {
                                Kz = offsets[t + 1] - offsets[t];
                                break;
                            }
                        }
                        total = total + (1 / Kz);
                        break;
                    }
                }
            }
            results[row * numNodes + col] = total;
        }
    }
}

__global__ void kernel_JI(bool *A, int *neighbors, int *offsets, float *results, int numNodes)
{
    // implementing the metric
    // Jaccard Index =  | N(u) \cap N(v) |/| N(u) \cup N(v) |
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // indexes of the neighbors of the first node
    int start1 = offsets[col];
    int end1 = offsets[col + 1];

    // indexes of the neighbors of the second node
    int start2 = offsets[row];
    int end2 = offsets[row + 1];

    int numberOfIntersections = 0;
    int numberOfUnion = 0;

    if (row < numNodes && col < numNodes)
    {
        // printf("nodes (%d, %d)\n", row, col);
        if (not A[row * numNodes + col])
        {
            numberOfUnion = end1 - start1;
            for (int i = start1; i < end1; ++i)
            {
                for (int j = start2; j < end2; ++j)
                {
                    if (neighbors[i] == neighbors[j])
                    {
                        numberOfIntersections++;
                        break;
                    }
                }

                bool found = false;
                for (int j = start2; j < end2; ++j)
                {
                    if (neighbors[i] == neighbors[j])
                    {
                        found = true;
                        break;
                    }
                }
                if (not found)
                {
                    numberOfUnion++;
                }
            }
            results[row * numNodes + col] = numberOfIntersections / numberOfUnion;
        }
    }
}

__global__ void kernel_SI(bool *A, int *neighbors, int *offsets, float *results, int numNodes)
{
    // implementing the metric
    // Sorensen Index = 2 | N(u) \cap N(v) |/(k_u + k_v)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // indexes of the neighbors of the first node
    int start1 = offsets[col];
    int end1 = offsets[col + 1];

    // indexes of the neighbors of the second node
    int start2 = offsets[row];
    int end2 = offsets[row + 1];

    int numberOfIntersections = 0;
    if (row < numNodes && col < numNodes)
    {
        // printf("nodes (%d, %d)\n", row, col);
        if (not A[row * numNodes + col])
        {
            for (int i = start1; i < end1; ++i)
            {
                for (int j = start2; j < end2; ++j)
                {
                    if (neighbors[i] == neighbors[j])
                    {
                        numberOfIntersections++;
                        break;
                    }
                }
            }

            // printf("(%d, %d) = %d\n", col, row, numberOfIntersections);
            results[row * numNodes + col] = 2 * numberOfIntersections / ((end1 - start1) + (end2 - start2));
        }
    }
}

__global__ void kernel_CN(bool *A, int *neighbors, int *offsets, float *results, int numNodes)
{
    // implementing the metric
    // Common Neighbors = | N(u) \cap N(v) |
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // indexes of the neighbors of the first node
    int start1 = offsets[col];
    int end1 = offsets[col + 1];

    // indexes of the neighbors of the second node
    int start2 = offsets[row];
    int end2 = offsets[row + 1];

    int numberOfIntersections = 0;
    if (row < numNodes && col < numNodes)
    {
        // printf("nodes (%d, %d)\n", row, col);
        if (not A[row * numNodes + col])
        {
            for (int i = start1; i < end1; ++i)
            {
                for (int j = start2; j < end2; ++j)
                {
                    if (neighbors[i] == neighbors[j])
                    {
                        numberOfIntersections++;
                        break;
                    }
                }
            }

            // printf("(%d, %d) = %d\n", col, row, numberOfIntersections);
            results[row * numNodes + col] = numberOfIntersections;
        }
    }
}

__global__ void kernel_HPI(bool *A, int *neighbors, int *offsets, float *results, int numNodes)
{
    // implementing the metric
    // Hub Promoted Index = | N(u) \cap N(v) |/min{k_u, k_v};
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // indexes of the neighbors of the first node
    int start1 = offsets[col];
    int end1 = offsets[col + 1];

    // indexes of the neighbors of the second node
    int start2 = offsets[row];
    int end2 = offsets[row + 1];

    int numberOfIntersections = 0;
    if (row < numNodes && col < numNodes)
    {
        // printf("nodes (%d, %d)\n", row, col);
        if (not A[row * numNodes + col])
        {
            for (int i = start1; i < end1; ++i)
            {
                for (int j = start2; j < end2; ++j)
                {
                    if (neighbors[i] == neighbors[j])
                    {
                        numberOfIntersections++;
                        break;
                    }
                }
            }

            if ((end2 - start2) < (end1 - start1))
            {
                results[row * numNodes + col] = numberOfIntersections / (end2 - start2);
            }
            else
            {
                results[row * numNodes + col] = numberOfIntersections / (end1 - start1);
            }
        }
    }
}

__global__ void kernel_HDI(bool *A, int *neighbors, int *offsets, float *results, int numNodes)
{
    // implementing the metric
    // Hub depressed Index = | N(u) \cap N(v) |/max{k_u, k_v};
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // indexes of the neighbors of the first node
    int start1 = offsets[col];
    int end1 = offsets[col + 1];

    // indexes of the neighbors of the second node
    int start2 = offsets[row];
    int end2 = offsets[row + 1];

    int numberOfIntersections = 0;
    if (row < numNodes && col < numNodes)
    {
        // printf("nodes (%d, %d)\n", row, col);
        if (not A[row * numNodes + col])
        {
            for (int i = start1; i < end1; ++i)
            {
                for (int j = start2; j < end2; ++j)
                {
                    if (neighbors[i] == neighbors[j])
                    {
                        numberOfIntersections++;
                        break;
                    }
                }
            }

            if ((end2 - start2) > (end1 - start1))
            {
                results[row * numNodes + col] = numberOfIntersections / (end2 - start2);
            }
            else
            {
                results[row * numNodes + col] = numberOfIntersections / (end1 - start1);
            }
        }
    }
}

__global__ void kernel_LLHNI(bool *A, int *neighbors, int *offsets, float *results, int numNodes)
{
    // implementing the metric
    // Local Leicht-Homle-Newman Index = | N(u) \cap N(v) |/(k_u* k_v);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // indexes of the neighbors of the first node
    int start1 = offsets[col];
    int end1 = offsets[col + 1];

    // indexes of the neighbors of the second node
    int start2 = offsets[row];
    int end2 = offsets[row + 1];

    int numberOfIntersections = 0;
    if (row < numNodes && col < numNodes)
    {
        // printf("nodes (%d, %d)\n", row, col);
        if (not A[row * numNodes + col])
        {
            for (int i = start1; i < end1; ++i)
            {
                for (int j = start2; j < end2; ++j)
                {
                    if (neighbors[i] == neighbors[j])
                    {
                        numberOfIntersections++;
                        break;
                    }
                }
            }

            results[row * numNodes + col] = numberOfIntersections / ((end2 - start2) * (end1 - start1));
        }
    }
}

__global__ void kernel_PA(bool *A, int *neighbors, int *offsets, float *results, int numNodes)
{
    // implementing the metric
    // Preferential attachment = k_u * k_v
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // indexes of the neighbors of the first node
    int start1 = offsets[col];
    int end1 = offsets[col + 1];

    // indexes of the neighbors of the second node
    int start2 = offsets[row];
    int end2 = offsets[row + 1];

    if (row < numNodes && col < numNodes)
    {

        if (not A[row * numNodes + col])
        {
            results[row * numNodes + col] = (end1 - start1) * (end2 - start2);
        }
    }
}

int main(int argc, char *argv[])
{
    auto start = std::chrono::high_resolution_clock::now();

    bool printing = false;

    // ***********************************   LP in cuda   ***************************
    // STEP ONE: read the edges list, and create the nodes list
    // ******************************************************************************

    // Check if a filename is passed
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <filename>\n";
        return 1;
    }
    // ******************************************************************************
    // number of lines == number of edges

    std::ifstream file(argv[1]); // Use the first argument as the file name
    std::string line;
    int totalEdges = 0;
    int totalNodes = 0;
    std::vector<int> Nodes;
    std::vector<std::string> edges;
    std::vector<std::string> test_edges;
    std::vector<std::string> train_edges;

    if (file.is_open())
    {
        // First pass to count the total number of edges
        while (getline(file, line))
        {
            totalEdges++;
        }

        // Reset the file pointer to the beginning of the file
        file.clear(); // Clear any error flags
        file.seekg(0, std::ios::beg);

        // Second pass to read and process each line

        while (getline(file, line))
        {
            std::istringstream iss(line);
            std::string node1, node2;
            if (!(iss >> node1 >> node2))
            {
                std::cerr << "Error reading line: " << line << '\n';
                continue; // Skip malformed lines
            }

            edges.push_back(line);

            bool found1 = false;
            bool found2 = false;

            // adding nodes
            for (const int &node : Nodes)
            {
                if (node == std::stoi(node1))
                    found1 = true;
                if (node == std::stoi(node2))
                    found2 = true;
                if (found1 && found2)
                    break;
            }

            if (not found1)
            {
                Nodes.push_back(std::stoi(node1));
                totalNodes++;
            }
            if (not found2)
            {
                Nodes.push_back(std::stoi(node2));
                totalNodes++;
            }
        }
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file: " << argv[1] << '\n';
        return 1;
    }

    if (printing)
    {
        for (const int &node : Nodes)
        {
            std::cout << "Node: " << node << std::endl;
        }
        std::cout << "number of Nodes: " << totalNodes << std::endl;
        std::cout << "number of Edges: " << totalEdges << std::endl;
    }

    // ******************************************************************************
    // STEP TWO: generate train and test edges lists
    // ******************************************************************************
    // Shuffle the array
    shuffleVector(edges);

    for (size_t i = 0; i < totalEdges; ++i)
    {
        if (i < totalEdges * 0.3)
        {
            test_edges.push_back(edges[i]);
        }
        else
        {
            train_edges.push_back(edges[i]);
        }
    }

    if (printing)
    {
        std::cout << "Number of elements in train_edges: " << train_edges.size() << std::endl;
        std::cout << "Number of elements in test_edges: " << test_edges.size() << std::endl;
    }

    // ******************************************************************************
    // STEP THREE: creating the node: [neighbors] for the train set
    // ******************************************************************************

    std::vector<std::list<int>> train_adjList(totalNodes);
    std::vector<std::list<int>> test_adjList(totalNodes);

    // first: the train edges
    for (const std::string &edge : train_edges)
    {
        std::istringstream iss(edge);
        std::string v, u;

        iss >> v >> u;

        int i = getIndex(Nodes, std::stoi(v));
        train_adjList[i].push_back(std::stoi(u));
    }

    // second: the test edges
    for (const std::string &edge : test_edges)
    {
        std::istringstream iss(edge);
        std::string v, u;

        iss >> v >> u;

        int i = getIndex(Nodes, std::stoi(v));
        test_adjList[i].push_back(std::stoi(u));
    }

    // ******************************************************************************
    // STEP FOUR: setting the adjacency matrix
    // ******************************************************************************
    // first: for the train set
    bool *A_t = (bool *)malloc(totalNodes * totalNodes * sizeof(bool));
    float *results_t = (float *)malloc(totalNodes * totalNodes * sizeof(float));

    // second: for the test set
    bool *A_p = (bool *)malloc(totalNodes * totalNodes * sizeof(bool));

    // initializing train and test data (A_t, results_t, A_p)
    for (int i = 0; i < totalNodes * totalNodes; i++)
    {
        A_t[i] = false;
        results_t[i] = 0;

        A_p[i] = false;
    };

    // setting the value of the adjacency matrix and results
    for (int i = 0; i < totalNodes - 1; i++)
    {
        for (int j = i + 1; j < totalNodes; j++)
        {
            // for the train set
            bool found = false;
            for (const int &neighbor : train_adjList[i])
            {
                if (Nodes[j] == neighbor)
                {
                    found = true;
                    break;
                }
            }
            if (found)
            {
                A_t[i * totalNodes + j] = true;
                A_t[j * totalNodes + i] = true;

                results_t[i * totalNodes + j] = 999;
                results_t[j * totalNodes + i] = 999;
            }
            // ----------------------------------------
            // for the test set
            found = false;
            for (const int &neighbor : test_adjList[i])
            {
                if (Nodes[j] == neighbor)
                {
                    found = true;
                    break;
                }
            }
            if (found)
            {
                A_p[i * totalNodes + j] = true;
                A_p[j * totalNodes + i] = true;
            }
        }
    }
    // ******************************************************************************
    // Convert (ONLY) the train information to flat array representation to be passed to CUDA
    // ******************************************************************************
    std::vector<int> neighbors, offsets;
    int totalNeighbors = 0;
    for (const auto &list : train_adjList)
    {
        offsets.push_back(totalNeighbors);
        neighbors.insert(neighbors.end(), list.begin(), list.end());
        totalNeighbors += list.size();
    }
    offsets.push_back(totalNeighbors); // Add the end offset

    // Stop timing
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    printf("Data preprocessing = %f\n", duration.count());

    printf("starting Cuda here...\n");
    // ******************************************************************************
    // Initialize time measurement
    float time_difference;
    cudaEvent_t startEvent, stopEvent;
    float runtime2, runtime3;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    // Allocate memory on GPU and copy data
    int *d_neighbors, *d_offsets, *d_Nodes;
    bool *d_A_t;
    float *d_results_t;

    // cuda memory allocation
    cudaMalloc(&d_neighbors, neighbors.size() * sizeof(int));
    cudaMalloc(&d_offsets, offsets.size() * sizeof(int));
    cudaMalloc(&d_A_t, totalNodes * totalNodes * sizeof(bool));
    cudaMalloc(&d_results_t, totalNodes * totalNodes * sizeof(float));
    cudaMalloc(&d_Nodes, totalNodes * sizeof(int));

    // int *U, *V, *Z;
    // cudaMalloc(&U, totalNodes * sizeof(int));
    // cudaMalloc(&V, totalNodes * sizeof(int));
    // cudaMalloc(&Z, totalNodes * sizeof(int));
    cudaEvent_t startEvent2, stopEvent2;
    cudaEventCreate(&startEvent2);
    cudaEventCreate(&stopEvent2);

    cudaEventRecord(startEvent2, 0);

    // cuda memory copying
    cudaMemcpy(d_Nodes, Nodes.data(), Nodes.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors, neighbors.data(), neighbors.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_t, A_t, totalNodes * totalNodes * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_results_t, results_t, totalNodes * totalNodes * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(stopEvent2, 0);
    cudaEventSynchronize(stopEvent2);
    cudaEventElapsedTime(&runtime2, startEvent2, stopEvent2);

    printf("Host to Device memory copying %10.4f ms\n", runtime2);

    // ******************************************************************************
    // Launch the kernel
    dim3 blockDim(8, 8);

    int blockX = (totalNodes + blockDim.x - 1) / blockDim.x;
    int blockY = (totalNodes + blockDim.y - 1) / blockDim.y;

    dim3 gridDim(blockX, blockY);

    // kernel_PA<<<gridDim, blockDim>>>(d_A_t, d_neighbors, d_offsets, d_results_t, totalNodes);
    kernel_CN<<<gridDim, blockDim>>>(d_A_t, d_neighbors, d_offsets, d_results_t, totalNodes);
    // kernel_HPI<<<gridDim, blockDim>>>(d_A_t, d_neighbors, d_offsets, d_results_t, totalNodes);
    // kernel_HDI<<<gridDim, blockDim>>>(d_A_t, d_neighbors, d_offsets, d_results_t, totalNodes);
    // kernel_LLHNI<<<gridDim, blockDim>>>(d_A_t, d_neighbors, d_offsets, d_results_t, totalNodes);

    // kernel_SI<<<gridDim, blockDim>>>(d_A_t, d_neighbors, d_offsets, d_results_t, totalNodes);
    // kernel_JI<<<gridDim, blockDim>>>(d_A_t, d_neighbors, d_offsets, d_results_t, totalNodes);
    // kernel_RA<<<gridDim, blockDim>>>(d_A_t, d_neighbors, d_offsets, d_results_t, d_Nodes, totalNodes);
    // kernel_AA<<<gridDim, blockDim>>>(d_A_t, d_neighbors, d_offsets, d_results_t, d_Nodes, totalNodes);

    cudaEvent_t startEvent3, stopEvent3;
    cudaEventCreate(&startEvent3);
    cudaEventCreate(&stopEvent3);

    cudaEventRecord(startEvent3, 0);
    cudaMemcpy(results_t, d_results_t, totalNodes * totalNodes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stopEvent3, 0);
    cudaEventSynchronize(stopEvent3);
    cudaEventElapsedTime(&runtime3, startEvent3, stopEvent3);

    printf("Device to Host memory copying %10.4f ms\n", runtime3);

    // Stop time measurement
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time_difference, startEvent, stopEvent);

    printf("Cuda is done here...\n");
    // ************************************************************************************

    std::vector<std::pair<float, int>> data;

    int t = 0;
    int e = 0;
    for (int i = 0; i < totalNodes; i++)
    {
        for (int j = 0; j < totalNodes; j++)
        {
            t = i * totalNodes + j;
            if (not A_t[t])
            {
                e = 0;
                if (A_p[t])
                {
                    e = 1;
                }
                data.push_back({results_t[t], e});
            }
        }
    }

    double auc = CalculateAUC(data);
    // double auc = CalculateAUPRC(data);
    printf("Nodes = %d\nEdges = %d\n", totalNodes, totalEdges);
    printf("%f ms total time.\nAUC was %.2f%\n", time_difference+runtime2+runtime3, auc);

    // Cleanup
    // Free GPU memory
    cudaFree(d_neighbors);
    cudaFree(d_offsets);
    cudaFree(d_A_t);
    cudaFree(d_results_t);
    cudaFree(d_Nodes);
    // Additionally, if used, free U, V, Z

    // Free CPU memory
    free(A_t);
    free(results_t);
    // Free A_p if it was allocated

    return 0;
}
