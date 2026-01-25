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
        for (auto const& kv : k) deg += kv.second;
        return deg;
    }

    void cleanup() {
        for (auto it = data.begin(); it != data.end();) {
            if (abs(it->second) < 1e-9) it = data.erase(it);
            else ++it;
        }
    }

    void add(const MultiPolynomial& other) {
        for (auto const& kv : other.data) data[kv.first] += kv.second;
        cleanup();
    }

    void subtract(const MultiPolynomial& other) {
        for (auto const& kv : other.data) data[kv.first] -= kv.second;
        cleanup();
    }

    MultiPolynomial multiply(const MultiPolynomial& other) const {
        MultiPolynomial res;
        for (auto const& a : data) {
            for (auto const& b : other.data) {
                TermKey newKey = a.first;
                for (auto const& kv : b.first) newKey[kv.first] += kv.second;
                res.data[newKey] += a.second * b.second;
            }
        }
        res.cleanup();
        return res;
    }

    double evaluate(const map<string, double>& varValues) const {
        double total = 0.0;
        for (auto const& term : data) {
            double termVal = term.second;
            for (auto const& kv : term.first) {
                auto it = varValues.find(kv.first);
                if (it != varValues.end()) termVal *= pow(it->second, kv.second);
                else if (kv.second > 0) { termVal = 0.0; break; }
            }
            total += termVal;
        }
        return total;
    }

    MultiPolynomial derivative(const string& varName) const {
        MultiPolynomial res;
        for (auto const& term : data) {
            auto it = term.first.find(varName);
            if (it != term.first.end()) {
                int exp = it->second;
                if (exp > 0) {
                    TermKey newKey = term.first;
                    if (exp == 1) newKey.erase(varName);
                    else newKey[varName] = exp - 1;
                    res.data[newKey] += term.second * exp;
                }
            }
        }
        res.cleanup();
        return res;
    }

    set<string> getVariables() const {
        set<string> vars;
        for (auto const& term : data) {
            for (auto const& kv : term.first) vars.insert(kv.first);
        }
        return vars;
    }

    string toString() const {
        if (data.empty()) return "0";

        struct TermItem { TermKey key; double coeff; int degree; };
        vector<TermItem> items;
        items.reserve(data.size());
        for (auto const& term : data) items.push_back({term.first, term.second, getTotalDegree(term.first)});

        sort(items.begin(), items.end(), [&](const TermItem& a, const TermItem& b) {
            if (a.degree != b.degree) return a.degree > b.degree;
            return a.key > b.key;
        });

        stringstream ss;
        bool first = true;
        for (auto const& item : items) {
            double c = item.coeff;
            const TermKey& k = item.key;

            if (!first && c > 0) ss << " + ";
            if (c < 0) ss << (first ? "-" : " - ");

            double absC = abs(c);
            if (abs(absC - 1.0) > 1e-9 || k.empty()) ss << absC;

            for (auto const& kv : k) {
                ss << kv.first;
                if (kv.second > 1) ss << "^" << kv.second;
            }
            first = false;
        }
        return ss.str();
    }
};

enum NodeType { NUMBER, VARIABLE, OPERATOR };
struct Node {
    NodeType type;
    char op;
    double val;
    string varName;
    bool isUnary;
    shared_ptr<Node> left, right;
    Node(double v) : type(NUMBER), op(0), val(v), isUnary(false) {}
    Node(const string& name) : type(VARIABLE), op(0), val(0), varName(name), isUnary(false) {}
    Node(char o, bool unary = false) : type(OPERATOR), op(o), val(0), isUnary(unary) {}
};

map<char, int> prec = {{'+', 1}, {'-', 1}, {'*', 2}, {'/', 2}, {'^', 3}, {'~', 4}};

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
        bool divisible = true;
        for (auto const& mkv : monomialKey) {
            newKey[mkv.first] -= mkv.second;
            if (newKey[mkv.first] == 0) newKey.erase(mkv.first);
        }
        res.data[newKey] += kv.second / monomialCoeff;
    }
    res.cleanup();
    return res;
}

string addExplicitMult(string s) {
    string res;
    res.reserve(s.size() * 2);
    for (int i = 0; i < (int)s.size(); i++) {
        if (isspace((unsigned char)s[i])) continue;
        if (!res.empty()) {
            char p = res.back();
            char c = s[i];
            bool pEnd = (isdigit((unsigned char)p) || isalpha((unsigned char)p) || p == ')');
            bool cStart = (isalpha((unsigned char)c) || c == '(');
            if (pEnd && cStart) res.push_back('*');
            else if ((isalpha((unsigned char)p) || p == ')') && isdigit((unsigned char)c)) res.push_back('*');
        }
        res.push_back(s[i]);
    }
    return res;
}

shared_ptr<Node> buildTree(const string& s) {
    stack<shared_ptr<Node>> nodes;
    stack<char> ops;
    bool unary = true;

    auto process = [&]() {
        char op = ops.top(); ops.pop();
        auto n = make_shared<Node>(op, op == '~');
        n->right = nodes.top(); nodes.pop();
        if (!n->isUnary) { n->left = nodes.top(); nodes.pop(); }
        nodes.push(n);
    };

    for (int i = 0; i < (int)s.size(); i++) {
        if (isdigit((unsigned char)s[i]) || s[i] == '.') {
            string v;
            while (i < (int)s.size() && (isdigit((unsigned char)s[i]) || s[i] == '.')) v.push_back(s[i++]);
            nodes.push(make_shared<Node>(stod(v)));
            i--;
            unary = false;
        } else if (isalpha((unsigned char)s[i])) {
            string v;
            while (i < (int)s.size() && isalpha((unsigned char)s[i])) v.push_back(s[i++]);
            nodes.push(make_shared<Node>(v));
            i--;
            unary = false;
        } else if (s[i] == '(') {
            ops.push('(');
            unary = true;
        } else if (s[i] == ')') {
            while (!ops.empty() && ops.top() != '(') process();
            if (!ops.empty()) ops.pop();
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

MultiPolynomial normalize(const shared_ptr<Node>& node) {
    if (!node) return MultiPolynomial();

    if (node->type == NUMBER) {
        MultiPolynomial p;
        p.data[{}] = node->val;
        p.cleanup();
        return p;
    }
    if (node->type == VARIABLE) {
        MultiPolynomial p;
        p.data[{{node->varName, 1}}] = 1.0;
        p.cleanup();
        return p;
    }

    MultiPolynomial L = normalize(node->left);
    MultiPolynomial R = normalize(node->right);

    if (node->isUnary) {
        MultiPolynomial z;
        z.subtract(R);
        return z;
    }
    if (node->op == '+') { L.add(R); return L; }
    if (node->op == '-') { L.subtract(R); return L; }
    if (node->op == '*') return L.multiply(R);
    if (node->op == '^') {
        double rv = R.evaluate({});
        long long expLL = (long long) llround(rv);
        if (abs(rv - (double)expLL) > 1e-9 || expLL < 0) return MultiPolynomial();
        int exp = (int)expLL;
        MultiPolynomial res;
        res.data[{}] = 1.0;
        for (int i = 0; i < exp; i++) res = res.multiply(L);
        res.cleanup();
        return res;
    }
    if (node->op == '/') {
        TermKey monomialKey;
        double monomialCoeff;
        if (isMonomial(R, monomialKey, monomialCoeff)) {
            return divideByMonomial(L, monomialKey, monomialCoeff);
        }
        else {
            return MultiPolynomial();
        }
    }
    return MultiPolynomial();
}

string findAllRoots(const MultiPolynomial& p, double rangeStart, double rangeEnd, const string& var) {
    vector<double> roots;
    MultiPolynomial d = p.derivative(var);
    double step = 0.2;

    for (double x0 = rangeStart; x0 < rangeEnd; x0 += step) {
        double f1 = p.evaluate({{var, x0}});
        double f2 = p.evaluate({{var, x0 + step}});
        if (f1 * f2 <= 0) {
            double x = x0 + step / 2.0;
            for (int i = 0; i < 40; i++) {
                double f = p.evaluate({{var, x}});
                double df = d.evaluate({{var, x}});
                if (abs(f) < 1e-12 || abs(df) < 1e-13) break;
                x = x - f / df;
            }
            if (abs(p.evaluate({{var, x}})) < 1e-8) {
                bool exists = false;
                for (double r : roots) if (abs(r - x) < 1e-4) { exists = true; break; }
                if (!exists) roots.push_back(x);
            }
        }
    }

    if (roots.empty()) return "Cannot find real solutions";
    stringstream ss;
    ss << "Have found " << roots.size() << " real solutions: ";
    for (double r : roots) ss << fixed << setprecision(4) << r << "  ";
    return ss.str();
}

bool hasValidParentheses(const string& s) {
    int bal = 0;
    for (char ch : s) {
        if (ch == '(') {
            ++bal;
        } else if (ch == ')') {
            if (bal == 0) return false; 
            --bal;
        }
    }
    return bal == 0; 
}

string run_expression_tree(string input) {
    if (input.empty() || input == "exit") return "";
    if (!input.empty() && input.back() == '=') input.pop_back();

     if (!hasValidParentheses(input)) {
        return "Expression_tree::run_expression_tree: Unmatched parentheses in the expression.";
    }

    string processed = addExplicitMult(input);
    size_t eqPos = processed.find('=');

    if (eqPos != string::npos) {
        MultiPolynomial pL = normalize(buildTree(processed.substr(0, eqPos)));
        MultiPolynomial pR = normalize(buildTree(processed.substr(eqPos + 1)));

        set<string> vars = pL.getVariables();
        set<string> varsR = pR.getVariables();
        vars.insert(varsR.begin(), varsR.end());

        if (vars.empty()) {
            double vL = pL.evaluate({});
            double vR = pR.evaluate({});
            string ok = (abs(vL - vR) < 1e-9) ? "is correct" : "is incorrect";
            return string("The expression: ") + to_string(vL) + " = " + to_string(vR) + " -> " + ok;
        }

        pL.subtract(pR);

        if (vars.size() == 1) {
            string vName = *vars.begin();
            return findAllRoots(pL, -1000, 1000, vName);
        }

        if (pL.data.empty()) return "They are equal.";
        return string("Result: ") + pL.toString() + " = 0";
    }

    MultiPolynomial p = normalize(buildTree(processed));
    set<string> vars = p.getVariables();
    if (vars.empty()) return string("Result: ") + to_string(p.evaluate({}));
    return string("Result: ") + p.toString();
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
    if (argc != 2){
        cout << "Usage: ./symbol_classifier <image_path>" << endl;
        return 1;
    }
    string weights_folder = "weights/";
    string image_path = string(argv[1]);
    string output_dir = "symbols";

    detect_and_save_symbols(image_path, output_dir);

    vector<string> classes = load_classes_from_file(weights_folder + "classes.txt");
    
    cout << "Loading CNN model..." << endl;
    CNN_model cnn;
    cnn.add_layer(new ConvolutionLayer(6, 5, 1, 2, 44, 44,1));
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
