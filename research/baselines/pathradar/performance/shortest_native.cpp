// Immutable integer graph; original PathRadar traversal and state publication order.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
struct Error {};
struct O {
    PyObject* p;
    explicit O(PyObject* v):p(v){if(!p)throw Error{};}
    ~O(){Py_XDECREF(p);}
    O(const O&)=delete;
    O(O&& v) noexcept:p(v.p){v.p=nullptr;}
    PyObject* release(){auto* v=p;p=nullptr;return v;}
    operator PyObject*()const{return p;}
};
void check(int value){if(value<0)throw Error{};}
using Id=uint32_t;
using Path=std::vector<Id>;
struct Hash {
    size_t operator()(const Path& p)const{
        size_t h=0;
        for(auto v:p)h^=size_t(v)+0x9e3779b9+(h<<6)+(h>>2);
        return h;
    }
};
struct Node {
    PyObject* key;
    long long numeric;
    std::array<std::vector<Id>,3> neighbors,membership;
    Node(PyObject* k,long long n):key(k),numeric(n){Py_INCREF(key);}
    Node(const Node&)=delete;
    Node(Node&& n)noexcept:key(n.key),numeric(n.numeric),neighbors(std::move(n.neighbors)),membership(std::move(n.membership)){n.key=nullptr;}
    ~Node(){Py_XDECREF(key);}
};
struct Graph {
    std::vector<Node> nodes;
    PyObject* index;
    Graph():index(PyDict_New()){if(!index)throw Error{};}
    ~Graph(){Py_XDECREF(index);}
    Id add(PyObject* key){
        auto* existing=PyDict_GetItemWithError(index,key);
        if(existing)return static_cast<Id>(PyLong_AsUnsignedLong(existing));
        if(PyErr_Occurred())throw Error{};
        O numeric(PyNumber_Long(key));
        auto value=PyLong_AsLongLong(numeric);
        if(value==-1&&PyErr_Occurred())throw Error{};
        Id id=static_cast<Id>(nodes.size());
        O identity(PyLong_FromUnsignedLong(id));
        check(PyDict_SetItem(index,key,identity));
        nodes.emplace_back(key,value);
        return id;
    }
    bool has(Id parent,int kind,Id neighbor)const{
        const auto& members=nodes[parent].membership[kind];
        return std::binary_search(members.begin(),members.end(),neighbor);
    }
};
constexpr const char* CAPSULE="pathradar.integer_graph.v1";
void destroy(PyObject* capsule){delete static_cast<Graph*>(PyCapsule_GetPointer(capsule,CAPSULE));}

PyObject* pack(PyObject*,PyObject* sample){
    try{
        auto graph=std::make_unique<Graph>();
        const char* names[]={"provider","peer","customer"};
        for(int kind=0;kind<3;kind++){
            O table(PyObject_GetAttrString(sample,names[kind]));
            if(!PyDict_Check(table.p)){PyErr_SetString(PyExc_TypeError,"relationships must be dictionaries");throw Error{};}
            Py_ssize_t pos=0;PyObject *key,*values;
            while(PyDict_Next(table,&pos,&key,&values)){
                auto id=graph->add(key);
                O iterator(PyObject_GetIter(values));
                while(auto* value=PyIter_Next(iterator)){
                    O owner(value);
                    auto neighbor=graph->add(value);
                    graph->nodes[id].neighbors[kind].push_back(neighbor);
                }
                if(PyErr_Occurred())throw Error{};
            }
        }
        for(auto& node:graph->nodes){
            node.membership=node.neighbors;
            for(auto& members:node.membership)std::sort(members.begin(),members.end());
        }
        O capsule(PyCapsule_New(graph.get(),CAPSULE,destroy));
        graph.release();
        return capsule.release();
    }catch(Error&){return nullptr;}
}

O path_object(const Graph& graph,const Path& path,bool tuple){
    O result(tuple?PyTuple_New(path.size()):PyList_New(path.size()));
    for(size_t i=0;i<path.size();i++){
        auto* key=graph.nodes[path[i]].key;Py_INCREF(key);
        if(tuple)PyTuple_SET_ITEM(result.p,i,key);else PyList_SET_ITEM(result.p,i,key);
    }
    return result;
}
bool smaller(const Graph& graph,Id parent,Id neighbor,const Path& worst){
    auto previous=worst[worst.size()-2],end=worst.back();
    if(graph.has(neighbor,2,parent)&&(graph.has(end,1,previous)||graph.has(end,0,previous)))return true;
    return graph.has(neighbor,1,parent)&&graph.has(end,0,previous);
}
struct Flag {Path path;int value;};

PyObject* shortest(PyObject*,PyObject* args){
    try{
        PyObject *capsule,*sample,*asn;
        int lengths_only=0;
        if(!PyArg_ParseTuple(args,"OOO|p",&capsule,&sample,&asn,&lengths_only))return nullptr;
        auto* graph=static_cast<Graph*>(PyCapsule_GetPointer(capsule,CAPSULE));
        if(!graph)throw Error{};
        auto* found=PyDict_GetItemWithError(graph->index,asn);
        if(!found){
            if(PyErr_Occurred())throw Error{};
            if(lengths_only){
                O output(PyDict_New()),key(PyTuple_Pack(2,asn,asn)),one(PyLong_FromLong(1));
                check(PyDict_SetItem(output,key,one));
                return output.release();
            }
            // An isolated source needs no graph mutation (the capsule stays fork-shared).
            O paths(PyObject_GetAttrString(sample,"AS_path")),worst(PyObject_GetAttrString(sample,"worst"));
            O flags(PyObject_GetAttrString(sample,"path_flag")),root(PyTuple_Pack(1,asn)),zero(PyLong_FromLong(0));
            O route(PySequence_List(root));
            check(PyObject_SetItem(paths,asn,route));check(PyObject_SetItem(worst,asn,route));
            check(PyObject_SetItem(flags,root,zero));
            Py_RETURN_NONE;
        }
        Id source=static_cast<Id>(PyLong_AsUnsignedLong(found));
        std::vector<Path> paths(graph->nodes.size()),worst(graph->nodes.size());
        std::vector<Id> discovery{source};
        paths[source]=worst[source]=Path{source};
        std::vector<Flag> flags{{Path{source},0}};
        std::unordered_map<Path,size_t,Hash> flag_index{{Path{source},0}};
        std::vector<Path> active{Path{source}};
        while(!active.empty()){
            std::vector<Path> next;
            for(int relation=0;relation<3;relation++){
                for(const auto& current:active){
                    auto parent=current.back();
                    auto state=flags[flag_index.at(current)].value;
                    for(auto neighbor:graph->nodes[parent].neighbors[relation]){
                        if(neighbor==source||(state!=0&&!graph->has(parent,2,neighbor)))continue;
                        bool shorter=paths[neighbor].empty()||current.size()+1<paths[neighbor].size();
                        bool better=worst[neighbor].empty()||smaller(*graph,parent,neighbor,worst[neighbor]);
                        if(!shorter&&!better)continue;
                        Path right=current;right.push_back(neighbor);
                        if(shorter){
                            if(paths[neighbor].empty())discovery.push_back(neighbor);
                            paths[neighbor]=right;
                        }
                        if(better)worst[neighbor]=right;
                        auto insert=flag_index.emplace(right,flags.size());
                        if(insert.second)flags.push_back(Flag{right,relation});
                        else flags[insert.first->second].value=relation;
                        next.push_back(std::move(right));
                    }
                }
            }
            std::stable_sort(next.begin(),next.end(),[graph](const Path& a,const Path& b){
                return graph->nodes[a.back()].numeric<graph->nodes[b.back()].numeric;
            });
            active=std::move(next);
            if(PyErr_CheckSignals()<0)throw Error{};
        }
        if(lengths_only){
            O output(PyDict_New());
            for(auto node:discovery){
                O key(PyTuple_Pack(2,asn,graph->nodes[node].key)),length(PyLong_FromSize_t(paths[node].size()));
                check(PyDict_SetItem(output,key,length));
            }
            return output.release();
        }
        O output(PyObject_GetAttrString(sample,"AS_path")),woutput(PyObject_GetAttrString(sample,"worst"));
        O foutput(PyObject_GetAttrString(sample,"path_flag"));
        for(auto node:discovery){
            O path=path_object(*graph,paths[node],false),wpath=path_object(*graph,worst[node],false);
            check(PyObject_SetItem(output,graph->nodes[node].key,path));
            check(PyObject_SetItem(woutput,graph->nodes[node].key,wpath));
        }
        for(const auto& entry:flags){
            O path=path_object(*graph,entry.path,true),flag(PyLong_FromLong(entry.value));
            check(PyObject_SetItem(foutput,path,flag));
        }
        Py_RETURN_NONE;
    }catch(Error&){return nullptr;}
}
PyMethodDef methods[]={
    {"pack",pack,METH_O,"Snapshot relationship sets in their original iteration order."},
    {"shortest_stream",shortest,METH_VARARGS,"Traverse an integer graph and publish original Python state."},
    {nullptr,nullptr,0,nullptr}
};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_pathradar_shortest",nullptr,-1,methods};
}
PyMODINIT_FUNC PyInit__pathradar_shortest(){return PyModule_Create(&module);}
