#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <vector>

#include "util.h"
#include "const.h"

int readToken(FILE *fp, int curr) {
  int c = fgetc(fp);
  if (c == EOF)
    return EOF;
  if (c == (int)'(' || c == (int)')' || c == (int)' ' || c == (int)'.')
    return curr;
  int val = (int)(c - (int)'0');
  return readToken(fp, curr == NUM_NOT_FOUND ? val : curr * 10 + val);
}

void reformatStore (int* store, int rows, int cols, FILE* handle) {
  fprintf(handle, "#hash(");
  for (int i = 0; i < rows; i++) {
    int bindings = store[i*cols + 0] - 1;
    fprintf (handle, "(%d . (", i);
    int j = 1;
    for (; j < bindings; j++) {
      fprintf(handle, "%d ", store[i*cols + j]);
    }
    if (j <= bindings) {
      fprintf(handle, "%d))", store[i*cols + bindings]);
    } else {
      fprintf(handle, "))");
    }
  }
  fprintf(handle, ")");
}

void printMatrix (int* matrix, int rows, int cols) {
  int i, j;
  printf("[");
  for (i = 0; i < rows; i++) {
      printf("[");
      for (j = 0; j < cols; j++) {
        printf("%u", *(matrix + i*cols + j));
          if (j < cols-1)
            printf(" ");
          }
        printf ("]");
    if(i < rows-1)
      printf("\n");
    if ((i + 1) % (((rows > 3) ? rows : 4) / 3) == 0)
      printf("\n");
  }
  printf("]\n");
}

void printStore (std::set<int> store[], int rows) {
  for (int i = 0; i < rows; i++) {
    std::cout << "Row: " << i << " > ";
    printSet(store[i]);
    std::cout << "\n";
  }
  std::cout << "\n\n";
}

void printArray(int *array, int size) {
  std::cout << "[";
  for (int i = 0; i < size; i++) {
    std::cout << array[i] << ", ";
  }
  std::cout << "]\n";
}

void printSet(std::set<int> s) {
  for (auto x : s) {
    std::cout << x << " ";
  }
}

void printQ(std::queue<int> q) {
  while (!q.empty()) {
    std::cout << q.front() << ", ";
    q.pop();
  }
  std::cout << "\n";
}

void printDeps(std::map<int, std::vector<int>> deps) {
  for (auto const &kv :deps) {
    std::cout << kv.first << " =>";
    for (auto const &i : kv.second)
      std::cout << " " << i;
    std::cout << std::endl;
  }
}

std::string exprId(int id, int lams) {
  if (id < lams) {
    return std::string("L").append(std::to_string(id));
  } else if (id < 2 * lams) {
    return std::string("A").append(std::to_string(id - lams));
  } else {
    return std::string("K").append(std::to_string(id - 2 * lams));
  }
}

void makeGraph(std::set<int> store[], int arg1Vec[], int arg2Vec[],
               int callFun[], std::map<int, std::vector<int>> &deps, int calls,
               int lams, int iter, int callSite) {
  fprintf(stdout, "\ndigraph {\n");
  fprintf(stdout, "label = \"Iteration: %d\";\n", iter);
  for (int i = 0; i < calls; i++) {
    fprintf(stdout,"%s [shape=\"oval\"];\n", exprId(i, lams).c_str());
  }
  for(int i = 0; i < calls; i++) {
    fprintf(stdout, "Call%d [shape = box];\n", i);
    fprintf(stdout, "Call%d -> %s [label = F];\n", i, exprId(callFun[i], lams).c_str());
    fprintf(stdout, "Call%d -> %s[label = A];\n", i, exprId(arg1Vec[i], lams).c_str());
    fprintf(stdout, "Call%d -> %s[label = K];\n", i, exprId(arg2Vec[i], lams).c_str());
  }
  for(int i = 0; i < lams*3; i++) {
    for(auto x : store[i]) {
      fprintf(stdout, "%s -> %s;\n", exprId(i , lams).c_str(), exprId(x, lams).c_str());
    }
  }
  fprintf(stdout, "Call%d [fillcolor = lightgray, color = red];\n", callSite);
  for (int i = 0; i < lams; i++) {
    fprintf(stdout, "L%d -> A%d [style = dashed];\n", i, i);
    fprintf(stdout, "L%d -> K%d [style = dashed];\n", i, i);
  }
  fprintf(stdout, "}\n");
}

// template <typename Iter, typename Q>
// void push_range(Q &q, Iter begin, Iter end) {
//   for (; begin != end; ++begin) {
//     q.push(*begin);
//   }
// }
void pushRange(std::queue<int> &q, std::vector<int>::iterator begin, std::vector<int>::iterator end) {
  for (; begin != end; ++begin) {
    q.push(*begin);
  }
}
