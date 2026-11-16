// Immutable native feature evidence and contiguous int64 matrices; no model changes.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
struct Error {};
struct O {
    PyObject* p;
    explicit O(PyObject* v):p(v){if(!v)throw Error{};}
    ~O(){Py_XDECREF(p);}
    O(const O&)=delete;
    PyObject* release(){auto* v=p;p=nullptr;return v;}
    operator PyObject*()const{return p;}
};
std::string string(PyObject* object){
    Py_ssize_t n=0;auto* s=PyUnicode_AsUTF8AndSize(object,&n);
    if(!s)throw Error{};
    return std::string(s,n);
}
int64_t integer(PyObject* value){
    auto result=PyLong_AsLongLong(value);
    if(result==-1&&PyErr_Occurred())throw Error{};
    return result;
}
using Id=uint32_t;
constexpr Id MISSING=std::numeric_limits<Id>::max();
uint64_t edge(Id a,Id b){return (uint64_t(a)<<32)|b;}
struct Node{std::string text;std::array<int64_t,3> degree{};bool core=false;};
struct Edge{int relation=-1;int64_t rank=-1,frequency=0;};
struct Evidence {
    std::unordered_map<std::string,Id> ids;
    std::vector<Node> nodes;
    std::unordered_map<uint64_t,Edge> edges;
    std::unordered_map<std::string,int64_t> previous,next;
    Id add(const std::string& text){
        auto found=ids.find(text);
        if(found!=ids.end())return found->second;
        if(nodes.size()>=MISSING){PyErr_SetString(PyExc_OverflowError,"too many AS nodes");throw Error{};}
        Id id=static_cast<Id>(nodes.size());
        ids.emplace(text,id);nodes.push_back(Node{text,{0,0,0},false});return id;
    }
    Id find(const std::string& text)const{auto p=ids.find(text);return p==ids.end()?MISSING:p->second;}
};
struct Distances{const Evidence* owner;std::string source;std::vector<int64_t> values;std::unordered_map<std::string,int64_t> extra;};
constexpr const char* EVIDENCE="pathradar.features.v1";
constexpr const char* DISTANCES="pathradar.feature_distances.v1";
void destroy_evidence(PyObject* object){delete static_cast<Evidence*>(PyCapsule_GetPointer(object,EVIDENCE));}
void destroy_distances(PyObject* object){delete static_cast<Distances*>(PyCapsule_GetPointer(object,DISTANCES));}

void frequencies(PyObject* sample,const char* name,std::unordered_map<std::string,int64_t>& target){
    O table(PyObject_GetAttrString(sample,name));
    target.reserve(PyDict_Size(table));
    Py_ssize_t pos=0;PyObject *key,*value;
    while(PyDict_Next(table,&pos,&key,&value))target.emplace(string(key),integer(value));
}
PyObject* pack(PyObject*,PyObject* sample){
    try{
        auto evidence=std::make_unique<Evidence>();
        const char* names[]={"provider","peer","customer"};
        for(int kind=0;kind<3;kind++){
            O table(PyObject_GetAttrString(sample,names[kind]));
            Py_ssize_t pos=0;PyObject *key,*values;
            while(PyDict_Next(table,&pos,&key,&values)){
                Id id=evidence->add(string(key));
                auto size=PyObject_Length(values);if(size<0)throw Error{};
                evidence->nodes[id].degree[kind]=size;
                O iterator(PyObject_GetIter(values));
                while(auto* raw=PyIter_Next(iterator)){O value(raw);evidence->add(string(value));}
                if(PyErr_Occurred())throw Error{};
            }
        }
        O core(PyObject_GetAttrString(sample,"core_cluster")),iterator(PyObject_GetIter(core));
        while(auto* raw=PyIter_Next(iterator)){O value(raw);auto id=evidence->add(string(value));evidence->nodes[id].core=true;}
        if(PyErr_Occurred())throw Error{};
        O order(PyObject_GetAttrString(sample,"linkorder"));
        Py_ssize_t pos=0;PyObject *key,*values;
        while(PyDict_Next(order,&pos,&key,&values)){
            auto source=evidence->add(string(key));
            O seq(PySequence_Fast(values,"link order must be a sequence"));
            for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(seq.p);i++){
                auto destination=evidence->add(string(PySequence_Fast_GET_ITEM(seq.p,i)));
                auto& item=evidence->edges[edge(source,destination)];
                if(item.rank==-1)item.rank=i;
            }
        }
        O relation(PyObject_GetAttrString(sample,"relationship"));pos=0;
        while(PyDict_Next(relation,&pos,&key,&values)){
            auto source=evidence->add(string(PyTuple_GET_ITEM(key,0)));
            auto destination=evidence->add(string(PyTuple_GET_ITEM(key,1)));
            auto name=string(values);
            evidence->edges[edge(source,destination)].relation=name=="c2p"?0:name=="p2p"?1:name=="p2c"?2:3;
        }
        O freq(PyObject_GetAttrString(sample,"linkfreq"));pos=0;
        while(PyDict_Next(freq,&pos,&key,&values)){
            auto source=evidence->add(string(PyTuple_GET_ITEM(key,0)));
            auto destination=evidence->add(string(PyTuple_GET_ITEM(key,1)));
            evidence->edges[edge(source,destination)].frequency=integer(values);
        }
        frequencies(sample,"prev_frequency",evidence->previous);
        frequencies(sample,"next_frequency",evidence->next);
        O capsule(PyCapsule_New(evidence.get(),EVIDENCE,destroy_evidence));
        evidence.release();return capsule.release();
    }catch(Error&){return nullptr;}catch(const std::bad_alloc&){return PyErr_NoMemory();}
}
PyObject* distances(PyObject*,PyObject* args){
    try{
        PyObject *capsule,*table,*asn;
        if(!PyArg_ParseTuple(args,"OOO",&capsule,&table,&asn))return nullptr;
        auto* evidence=static_cast<Evidence*>(PyCapsule_GetPointer(capsule,EVIDENCE));if(!evidence)throw Error{};
        auto result=std::make_unique<Distances>();result->owner=evidence;result->source=string(asn);
        result->values.resize(evidence->nodes.size(),-1);
        Py_ssize_t pos=0;PyObject *key,*value;
        while(PyDict_Next(table,&pos,&key,&value)){
            if(string(PyTuple_GET_ITEM(key,0))!=result->source)continue;
            auto target=string(PyTuple_GET_ITEM(key,1));auto id=evidence->find(target);
            auto length=integer(value);
            if(length<0){PyErr_SetString(PyExc_ValueError,"negative shortest-path length");throw Error{};}
            if(id==MISSING)result->extra.emplace(target,length);else result->values[id]=length;
        }
        O capsule_result(PyCapsule_New(result.get(),DISTANCES,destroy_distances));
        result.release();return capsule_result.release();
    }catch(Error&){return nullptr;}catch(const std::bad_alloc&){return PyErr_NoMemory();}
}

std::array<int64_t,2> support(const std::unordered_map<std::string,int64_t>& table,
                            const std::vector<std::string>& nodes,bool suffix){
    std::string joined;std::vector<size_t> starts,ends;
    for(const auto& node:nodes){
        if(!starts.empty())joined+='|';
        starts.push_back(joined.size());joined+=node;ends.push_back(joined.size());
    }
    for(size_t length=nodes.size();length>1;length--){
        auto key=suffix?joined.substr(starts[nodes.size()-length]):joined.substr(0,ends[length-1]);
        auto found=table.find(key);
        if(found!=table.end())return {static_cast<int64_t>(length),found->second};
    }
    return {1,0};
}

PyObject* features(PyObject*,PyObject* args){
    try{
        PyObject *capsule,*dist_capsule,*raw;
        if(!PyArg_ParseTuple(args,"OOO",&capsule,&dist_capsule,&raw))return nullptr;
        auto* evidence=static_cast<Evidence*>(PyCapsule_GetPointer(capsule,EVIDENCE));if(!evidence)throw Error{};
        auto* lengths=static_cast<Distances*>(PyCapsule_GetPointer(dist_capsule,DISTANCES));if(!lengths)throw Error{};
        if(lengths->owner!=evidence){PyErr_SetString(PyExc_ValueError,"distance and evidence identities differ");throw Error{};}
        O paths(PySequence_Fast(raw,"paths must be a sequence"));
        auto count=PySequence_Fast_GET_SIZE(paths.p);
        std::vector<int64_t> matrix;matrix.reserve(count*20);
        int columns=0;
        for(Py_ssize_t row=0;row<count;row++){
            auto* path=PySequence_Fast_GET_ITEM(paths.p,row);
            if(!PyTuple_Check(path)||PyTuple_GET_SIZE(path)<2){PyErr_SetString(PyExc_ValueError,"expected AS tuple of length >=2");throw Error{};}
            size_t n=PyTuple_GET_SIZE(path);std::vector<Id> ids;std::vector<std::string> nodes;
            ids.reserve(n);nodes.reserve(n);
            for(size_t i=0;i<n;i++){nodes.push_back(string(PyTuple_GET_ITEM(path,i)));ids.push_back(evidence->find(nodes.back()));}
            if(nodes.front()!=lengths->source){PyErr_SetString(PyExc_ValueError,"path source differs from shortest-path source");throw Error{};}
            auto distance=[&](size_t at,int64_t fallback){
                if(ids[at]!=MISSING){auto value=lengths->values[ids[at]];return value<0?fallback:value;}
                auto found=lengths->extra.find(nodes[at]);return found==lengths->extra.end()?fallback:found->second;
            };
            auto get_edge=[&](size_t a,size_t b)->const Edge*{
                if(ids[a]==MISSING||ids[b]==MISSING)return nullptr;
                auto found=evidence->edges.find(edge(ids[a],ids[b]));return found==evidence->edges.end()?nullptr:&found->second;
            };
            bool core=ids.back()!=MISSING&&evidence->nodes[ids.back()].core;
            int width=core?20:17;
            if(columns&&columns!=width){PyErr_SetString(PyExc_ValueError,"mixed core/shell feature widths");throw Error{};}
            columns=width;auto slen=distance(n-1,n);
            matrix.push_back(n);matrix.push_back(slen);
            for(auto at:{n-1,n-2})for(int kind=0;kind<3;kind++)matrix.push_back(ids[at]==MISSING?0:evidence->nodes[ids[at]].degree[kind]);
            if(core){auto values=support(evidence->next,nodes,true);matrix.insert(matrix.end(),values.begin(),values.end());auto* e=get_edge(n-1,n-2);matrix.push_back(e?e->frequency:0);}
            auto* reverse=get_edge(n-1,n-2);auto* forward=get_edge(n-2,n-1);
            matrix.push_back(reverse?reverse->rank:-1);matrix.push_back(forward?forward->rank:-1);
            auto values=support(evidence->previous,nodes,false);matrix.insert(matrix.end(),values.begin(),values.end());
            std::array<int64_t,3> relations{};
            for(size_t i=0;i<n-1;i++){
                auto* e=get_edge(i,i+1);
                if(!e||e->relation==-1){O key(PyTuple_Pack(2,PyTuple_GET_ITEM(path,i),PyTuple_GET_ITEM(path,i+1)));PyErr_SetObject(PyExc_KeyError,key);throw Error{};}
                if(e->relation<3)relations[e->relation]++;
            }
            matrix.insert(matrix.end(),relations.begin(),relations.end());
            matrix.push_back(static_cast<int64_t>(n)-slen);matrix.push_back(static_cast<int64_t>(n)-1-distance(n-2,n-1));
        }
        O buffer(PyBytes_FromStringAndSize(reinterpret_cast<const char*>(matrix.data()),matrix.size()*sizeof(int64_t)));
        return Py_BuildValue("Nni",buffer.release(),count,columns);
    }catch(Error&){return nullptr;}catch(const std::bad_alloc&){return PyErr_NoMemory();}
}
PyMethodDef methods[]={
    {"pack",pack,METH_O,"Snapshot feature evidence into immutable native structures."},
    {"distances",distances,METH_VARARGS,"Encode a destination's shortest-path table once."},
    {"features",features,METH_VARARGS,"Construct a contiguous int64 feature matrix in original column order."},
    {nullptr,nullptr,0,nullptr}
};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_pathradar_features",nullptr,-1,methods};
}
PyMODINIT_FUNC PyInit__pathradar_features(){return PyModule_Create(&module);}
