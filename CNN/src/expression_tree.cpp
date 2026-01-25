#include <bits/stdc++.h>
#include "CNN.h"
#include "ConvolutionLayer.h"
#include "ReluLayer.h"
#include "Pool.h"
#include "DropOutLayer.h"
#include "FC.h"
#include "read_mnist.h"
#include "read_jpg.h"
#include "symbols.h"
#include "SoftmaxLayer.h"

using namespace std;
namespace fs = filesystem;

typedef map<string, int> TermKey;
struct MultiPolynomial {
    map<TermKey, double> data;

    int getTotalDegree(const TermKey& k) const {
        int deg = 0;
        for (auto const& [var, exp] : k) deg += exp;
        return deg;
    }

    void cleanup() {
        for (auto it = data.begin(); it != data.end(); ) {
            if (abs(it->second) < 1e-9) it = data.erase(it);
            else ++it;
        }
    }

    void add(const MultiPolynomial& other) {
        for (auto const& [key, coeff] : other.data) data[key] += coeff;
        cleanup();
    }

    void subtract(const MultiPolynomial& other) {
        for (auto const& [key, coeff] : other.data) data[key] -= coeff;
        cleanup();
    }

    MultiPolynomial multiply(const MultiPolynomial& other) {
        MultiPolynomial res;
        for (auto const& [key1, c1] : data) {
            for (auto const& [key2, c2] : other.data) {
                TermKey newKey = key1;
                for (auto const& [var, exp] : key2) newKey[var] += exp;
                res.data[newKey] += c1 * c2;
            }
        }
        res.cleanup();
        return res;
    }

    double evaluate(const map<string, double>& varValues) const {
        double total = 0;
        for (auto const& [key, coeff] : data) {
            double termVal = coeff;
            for (auto const& [var, exp] : key) {
                if (varValues.count(var)) termVal *= pow(varValues.at(var), exp);
                else if (exp > 0) termVal = 0;
            }
            total += termVal;
        }
        return total;
    }

    MultiPolynomial derivative(string varName) const {
        MultiPolynomial res;
        for (auto const& [key, coeff] : data) {
            if (key.count(varName)) {
                int exp = key.at(varName);
                if (exp > 0) {
                    TermKey newKey = key;
                    if (exp == 1) newKey.erase(varName);
                    else newKey[varName] = exp - 1;
                    res.data[newKey] += coeff * exp;
                }
            }
        }
        return res;
    }

    set<string> getVariables() const {
        set<string> vars;
        for (auto const& [key, coeff] : data) {
            for (auto const& [var, exp] : key) vars.insert(var);
        }
        return vars;
    }

    string toString() const {
        if (data.empty()) return "0";

        struct TermItem { TermKey key; double coeff; int degree; };
        vector<TermItem> items;
        for (auto const& [key, coeff] : data) {
            items.push_back({key, coeff, getTotalDegree(key)});
        }

        sort(items.begin(), items.end(), [](const TermItem& a, const TermItem& b) {
            if (a.degree != b.degree) return a.degree > b.degree;
            return a.key > b.key;
        });

        stringstream ss;
        bool first = true;
        for (auto const& item : items) {
            double c = item.coeff;
            TermKey k = item.key;

            if (!first && c > 0) ss << " + ";
            if (c < 0) ss << (first ? "-" : " - ");

            double absC = abs(c);
            if (abs(absC - 1.0) > 1e-9 || k.empty()) {
                ss << absC;
            }

            for (auto const& [var, exp] : k) {
                ss << var;
                if (exp > 1) ss << "^" << exp;
            }
            first = false;
        }
        return ss.str();
    }
};

enum NodeType { NUMBER, VARIABLE, OPERATOR };
struct Node {
    NodeType type; char op; double val; string varName; bool isUnary;
    shared_ptr<Node> left, right;
    Node(double v) : type(NUMBER), val(v), op(0), isUnary(false) {}
    Node(string name) : type(VARIABLE), varName(name), val(0), op(0), isUnary(false) {}
    Node(char o, bool u = false) : type(OPERATOR), op(o), val(0), isUnary(u) {}
};

map<char, int> prec = {{'+', 1}, {'-', 1}, {'*', 2}, {'/', 2}, {'^', 3}, {'~', 4}};

string preprocess(string s) {
    string res = "";
    for (int i = 0; i < (int)s.length(); i++) {
        if (isspace((unsigned char)s[i])) continue;
        if (i > 0 && !res.empty()) {
            char p = res.back(), c = s[i];
            bool pEnd = (isdigit((unsigned char)p) || isalpha((unsigned char)p) || p == ')');
            bool cStart = (isalpha((unsigned char)c) || c == '(');
            if (pEnd && cStart) res += '*';
            else if ((isalpha((unsigned char)p) || p == ')') && isdigit((unsigned char)c)) res += '*';
        }
        res += s[i];
    }
    return res;
}

shared_ptr<Node> buildTree(string s) {
    if (s.empty()) return nullptr;

    if (s.front() == '(' && s.back() == ')') {
        return buildTree(s.substr(1, s.length() - 2));
    }

    stack<shared_ptr<Node>> nodes;
    stack<char> ops;
    bool unary = true;

    auto process = [&]() {
        if (ops.empty() || nodes.empty()) return;
        char op = ops.top(); ops.pop();
        auto n = make_shared<Node>(op, op == '~');
        n->right = nodes.top(); nodes.pop();
        if (!n->isUnary && !nodes.empty()) { n->left = nodes.top(); nodes.pop(); }
        nodes.push(n);
    };

    for (int i = 0; i < (int)s.length(); i++) {
        if (isdigit((unsigned char)s[i]) || s[i] == '.') {
            string v;
            while (i < (int)s.length() && (isdigit((unsigned char)s[i]) || s[i] == '.')) v += s[i++];
            nodes.push(make_shared<Node>(stod(v)));
            i--;
            unary = false;
        } else if (isalpha((unsigned char)s[i])) {
            string v;
            while (i < (int)s.length() && isalpha((unsigned char)s[i])) v += s[i++];
            nodes.push(make_shared<Node>(v));
            i--;
            unary = false;
        } else if (s[i] == '(') {
            ops.push('(');
            unary = true;
        } else if (s[i] == ')') {
            while (!ops.empty() && ops.top() != '(') process();
            if (!ops.empty() && ops.top() == '(') ops.pop();
            unary = false;
        } else {
            char c = s[i];
            if (c == '-' && unary) c = '~';
            while (!ops.empty() && ops.top() != '(' &&
                   (prec[ops.top()] > prec[c] || (prec[ops.top()] == prec[c] && c != '^' && c != '~'))) {
                process();
            }
            ops.push(c);
            unary = true;
        }
    }
    while (!ops.empty()) process();
    return nodes.empty() ? nullptr : nodes.top();
}

bool hasVariables(shared_ptr<Node> node) {
    if (!node) return false;
    if (node->type == VARIABLE) return true;
    return hasVariables(node->left) || hasVariables(node->right);
}

bool isMonomial(const MultiPolynomial& p, TermKey& key, double& coeff) {
    if (p.data.size() != 1) return false;
    auto it = p.data.begin();
    key = it->first;
    coeff = it->second;
    return true;
}

MultiPolynomial divideByMonomial(const MultiPolynomial& p, const TermKey& monomialKey, double monomialCoeff) {
    MultiPolynomial res;
    if (abs(monomialCoeff) < 1e-12) return res;
    for (auto const& kv : p.data) {
        TermKey newKey = kv.first;
        for (auto const& mkv : monomialKey) {
            newKey[mkv.first] -= mkv.second;
            if (newKey[mkv.first] == 0) newKey.erase(mkv.first);
        }
        res.data[newKey] += kv.second / monomialCoeff;
    }
    res.cleanup();
    return res;
}

MultiPolynomial normalize(shared_ptr<Node> node) {
    if (!node) return MultiPolynomial();
    if (node->type == NUMBER) { MultiPolynomial p; p.data[{}] = node->val; p.cleanup(); return p; }
    if (node->type == VARIABLE) { MultiPolynomial p; p.data[{{node->varName, 1}}] = 1.0; p.cleanup(); return p; }

    MultiPolynomial L = normalize(node->left), R = normalize(node->right);

    if (node->isUnary) { MultiPolynomial z; z.subtract(R); return z; }
    if (node->op == '+') { L.add(R); return L; }
    if (node->op == '-') { L.subtract(R); return L; }
    if (node->op == '*') return L.multiply(R);
    if (node->op == '/') {
        if (R.getVariables().empty()) {
            double div = R.evaluate({});
            if (abs(div) < 1e-9) {
                cout << "Error: Division by zero." << endl;
                return MultiPolynomial();
            }
            MultiPolynomial res = L;
            for (auto& [key, coeff] : res.data) coeff /= div;
            res.cleanup();
            return res;
        } else {
            TermKey monomialKey;
            double monomialCoeff;
            if (isMonomial(R, monomialKey, monomialCoeff)) {
                if (abs(monomialCoeff) < 1e-12) {
                    cout << "Error: Division by zero." << endl;
                    return MultiPolynomial();
                }
                return divideByMonomial(L, monomialKey, monomialCoeff);
            } else {
                cout << "Error: Polynomial division (rational functions) is not supported." << endl;
                return MultiPolynomial();
            }
        }
    }
    if (node->op == '^') {
        double rv = R.evaluate({});
        long long expLL = (long long) llround(rv);
        if (abs(rv - (double)expLL) > 1e-9 || expLL < 0) return MultiPolynomial();
        int exp = (int)expLL;
        MultiPolynomial res; res.data[{}] = 1.0;
        for (int i = 0; i < exp; i++) res = res.multiply(L);
        res.cleanup();
        return res;
    }
    return MultiPolynomial();
}

string findAllRootsStr(const MultiPolynomial& p, double rangeStart, double rangeEnd, const string& var) {
    vector<double> roots;
    MultiPolynomial d = p.derivative(var);
    double step = 0.2;

    for (double x0 = rangeStart; x0 < rangeEnd; x0 += step) {
        double f1 = p.evaluate({{var, x0}}), f2 = p.evaluate({{var, x0 + step}});
        if (f1 * f2 <= 0) {
            double x = x0 + step / 2.0;
            for (int i = 0; i < 40; i++) {
                double f = p.evaluate({{var, x}}), df = d.evaluate({{var, x}});
                if (abs(f) < 1e-12 || abs(df) < 1e-13) break;
                x = x - f / df;
            }
            if (abs(p.evaluate({{var, x}})) < 1e-8) {
                bool exists = false;
                for (double r : roots) if (abs(r - x) < 1e-4) exists = true;
                if (!exists) roots.push_back(x);
            }
        }
    }

    if (roots.empty()) return "Cannot find real solutions";

    stringstream ss;
    ss << "Have found " << roots.size() << " real solutions: ";
    for (double r : roots) ss << fixed << setprecision(6) << r << "  ";
    return ss.str();
}

bool checkParentheses(const string& s) {
    int balance = 0;
    for (char c : s) {
        if (c == '(') balance++;
        else if (c == ')') {
            balance--;
            if (balance < 0) return false;
        }
    }
    return balance == 0;
}

string run_expression_tree(string input) {
    if (input.empty() || input == "exit") return "";
    if (!input.empty() && input.back() == '=') input.pop_back();

    size_t eqPos = input.find('=');

    if (eqPos != string::npos) {
        string partL = input.substr(0, eqPos);
        string partR = input.substr(eqPos + 1);

        if (!checkParentheses(partL) || !checkParentheses(partR)) {
            return "Error: Mismatched parentheses in equation sides!";
        }

        auto rootL = buildTree(preprocess(partL));
        auto rootR = buildTree(preprocess(partR));

        bool isEquation = hasVariables(rootL) || hasVariables(rootR);

        MultiPolynomial pL = normalize(rootL);
        MultiPolynomial pR = normalize(rootR);

        set<string> varsL = pL.getVariables();
        set<string> varsR = pR.getVariables();
        varsL.insert(varsR.begin(), varsR.end());

        if (varsL.empty()) {
            double vL = pL.evaluate({}), vR = pR.evaluate({});
            if (abs(vL - vR) < 1e-9) {
                if (isEquation) return "Result: The equation is an identity.";
                else return string("The expression: ") + to_string(vL) + " = " + to_string(vR) + " -> is correct";
            } else {
                if (isEquation) return "Result: No solution.";
                return string("The expression: ") + to_string(vL) + " = " + to_string(vR) + " -> is incorrect";
            }
        }

        else if (varsL.size() == 1) {
            string vName = *varsL.begin();
            string ss;
            ss = "The equation: " + pL.toString() + " = " + pR.toString() + "\n";
            pL.subtract(pR);
            if (pL.data.empty()) {
                ss += "Result: The equation is an identity";
                return ss;
            } else if (pL.getVariables().empty()) {
                double val = pL.evaluate({});
                if (abs(val) > 1e-9) ss += "Result: No solution";
                else ss += "Result: The equation is an identity";
                return ss;
            } else {
                ss += findAllRootsStr(pL, -1000, 1000, vName);
                return ss;
            }
        }
        else {
            pL.subtract(pR);
            if (pL.data.empty()) return "They are equal.";
            return string("Result: ") + pL.toString() + " = 0";
        }
    }
    else{
        if (!checkParentheses(input)) {
            return "Error: Mismatched parentheses!";
        }

        MultiPolynomial p = normalize(buildTree(preprocess(input)));
        set<string> vars = p.getVariables();
        if (!vars.empty()){
            string res_1 = "Result: " + p.toString() + "\n";
            map<string, double> vals;
            cout << "Input the values of variables:" << endl;
            for (string v : vars) {
                cout << v << " = "; cin >> vals[v];
            }
            res_1 += "Result: " + to_string(p.evaluate(vals)) + "\n";
            cin.ignore(1000, '\n');
            return res_1;
        }
        else {
            return "Result: " + to_string(p.evaluate({})) + "\n";
        }
    }
}

vector<float> load_vector_from_file(const string& filename) {
    ifstream ifs(filename);
    if (!ifs) throw runtime_error("Expression_tree::load_vector_from_file: Could not open file: " + filename);

    vector<float> data;
    string line;
    while (getline(ifs, line)) {
        if (line.empty()) continue;
        data.push_back(stof(line));
    }
    return data;
}

vector<string> load_classes_from_file(const string& filename) {
    ifstream ifs(filename);
    if (!ifs) throw runtime_error("Expression_tree::load_classes_from_file: Could not open file: " + filename);

    vector<string> classes;
    string line;
    while (getline(ifs, line)) {
        if (line.empty()) continue;
        classes.push_back(line);
    }
    return classes;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: ./cnn_solver <image_path>" << endl;
        return 1;
    }

    string weights_folder = "weights/";
    string image_path = string(argv[1]);
    string output_dir = "symbols";

    detect_and_save_symbols(image_path, output_dir);

    vector<string> classes = load_classes_from_file(weights_folder + "classes.txt");

    cout << "Loading CNN model..." << endl;
    CNN_model cnn;
    cnn.add_layer(new ConvolutionLayer(6, 5, 1, 2, 44, 44, 1));
    cnn.layers[0]->load_filters(load_vector_from_file(weights_folder + "conv1_filters.txt"));
    cnn.layers[0]->load_biases(load_vector_from_file(weights_folder + "conv1_biases.txt"));
    vector<int> out = cnn.get_output_size();
    cnn.add_layer(new ReluLayer(out[0], out[1], out[2]));
    out = cnn.get_output_size();
    cnn.add_layer(new PoolLayer(out[0], out[1], out[2], 2, 2));
    out = cnn.get_output_size();
    cnn.add_layer(new ConvolutionLayer(16, 5, 1, 2, out[1], out[2], out[0]));
    cnn.layers[3]->load_filters(load_vector_from_file(weights_folder + "conv2_filters.txt"));
    cnn.layers[3]->load_biases(load_vector_from_file(weights_folder + "conv2_biases.txt"));
    out = cnn.get_output_size();
    cnn.add_layer(new ReluLayer(out[0], out[1], out[2]));
    out = cnn.get_output_size();
    cnn.add_layer(new PoolLayer(out[0], out[1], out[2], 2, 2));
    out = cnn.get_output_size();
    cnn.add_layer(new FCLayer(out[0], out[1], out[2], 120));
    cnn.layers[6]->load_weights(load_vector_from_file(weights_folder + "fc1_weights.txt"));
    cnn.layers[6]->load_biases(load_vector_from_file(weights_folder + "fc1_biases.txt"));
    out = cnn.get_output_size();
    cnn.add_layer(new ReluLayer(out[0], out[1], out[2]));
    cnn.add_layer(new FCLayer(out[0], out[1], out[2], 84));
    cnn.layers[8]->load_weights(load_vector_from_file(weights_folder + "fc2_weights.txt"));
    cnn.layers[8]->load_biases(load_vector_from_file(weights_folder + "fc2_biases.txt"));
    out = cnn.get_output_size();
    cnn.add_layer(new ReluLayer(out[0], out[1], out[2]));
    out = cnn.get_output_size();
    cnn.add_layer(new FCLayer(out[0], out[1], out[2], (int)classes.size()));
    cnn.layers[10]->load_weights(load_vector_from_file(weights_folder + "fc3_weights.txt"));
    cnn.layers[10]->load_biases(load_vector_from_file(weights_folder + "fc3_biases.txt"));
    out = cnn.get_output_size();
    cnn.add_layer(new SoftmaxLayer(out[0]));

    cout << "Classifying symbols..." << endl;

    image_data data;
    data.read_data(output_dir);
    vector<vector<float>> images = data.get_images();

    string equation;
    for (auto img_it = images.begin(); img_it != images.end(); ++img_it) {
        const vector<float>& outp = cnn.predict(*img_it, false);

        int pred = 0;
        float best = outp[0];
        for (int i = 1; i < (int)outp.size(); i++) {
            if (outp[i] > best) {
                best = outp[i];
                pred = i;
            }
        }

        if (classes[pred] == "dot") equation += "*";
        else if (classes[pred] == "forward_slash") equation += "/";
        else if (classes[pred] == "plus") equation += "+";
        else if (classes[pred] == "minus") equation += "-";
        else equation += classes[pred];
    }

    cout << "Predicted equation: " << equation << endl;
    cout << "Solving equation..." << endl;

    string result = run_expression_tree(equation);
    cout << result << endl << endl;

    return 0;
}
