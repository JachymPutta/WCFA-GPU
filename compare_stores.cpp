#include <bits/types/FILE.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdio.h>
#include <string>
#include <vector>

void readLine(char* fileName, std::string &res) {
  std::ifstream file(fileName);
  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
      res = line.c_str();
    }
    file.close();
  }
}

void findAllMatches(std::string s, std::regex pattern, std::vector<std::string>& v){

  auto matchBegin = std::sregex_iterator(s.begin(), s.end(), pattern);
  auto matchEnd = std::sregex_iterator();

  for (std::sregex_iterator i = matchBegin; i != matchEnd; ++i) {

    std::smatch match = *i;
    std::string match_str = match.str();

    v.push_back(match_str);
  }
}


int main(int argc, char** argv){
  if (argc <= 2){
    fprintf(stderr, "Please input stores to compare !\n");
    return 1;
  }

  int same = 0;
  char* f1 = argv[1];
  char *f2 = argv[2];
  std::string s1;
  std::string s2;
  std::regex pattern ("\\(\[0-9]+ . \\((\[0-9] ?)*\\)\\)");
  std::vector<std::string> v1;
  std::vector<std::string> v2;

  readLine(f1, s1);
  readLine(f2, s2);

  s1 = s1.substr(6);
  s2 = s2.substr(6);
  s1.pop_back();
  s2.pop_back();

  findAllMatches(s1, pattern, v1);
  findAllMatches(s2, pattern, v2);

  if (v1.size() != v2.size()){
    fprintf(stderr, "Stores of different size for %s %i and %s %i\n",f1,int(v1.size()), f2, int(v2.size()));
    return 1;
  }

  fprintf(stderr, "Find differences between %s and %s:\n", f1, f2);
  for (int i = 0; i < v1.size(); i++){
    if (v1[i] != v2[i]){
      same++;
      fprintf(stderr, "Row %i not equal:\n %s: %s   %s: %s\n",
              i, f1, v1[i].c_str(), f2, v2[i].c_str());
    }
  }
  if (!same){
    fprintf(stderr, "%s %s - no differences found.\n", f1 ,f2);
  } else {
    fprintf(stderr, "%s, %s - %d differences found.\n", f1, f2, same); 
  }
  return 0;
}
