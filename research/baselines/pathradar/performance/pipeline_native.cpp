// Reuse verified feature semantics; add immutable ordered adjacency and cached comparisons.
#define PyInit__pathradar_features PyInit__pathradar_features_embedded
#include "features_native.cpp"
#undef PyInit__pathradar_features
#include <algorithm>

namespace {
void checked(int value){if(value<0)throw Error{};}
struct Adjacency {
    const Evidence* owner;
    PyObject* evidence_capsule;
    std::vector<std::array<std::vector<Id>,3>> neighbors;
    std::vector<PyObject*> keys;
    ~Adjacency(){for(auto* key:keys)Py_XDECREF(key);Py_XDECREF(evidence_capsule);}
};
constexpr const char* ADJACENCY="pathradar.pipeline_adjacency.v1";
void destroy_adjacency(PyObject* object){delete static_cast<Adjacency*>(PyCapsule_GetPointer(object,ADJACENCY));}
PyObject* pack_adjacency(PyObject*,PyObject* args){
    try{
        PyObject *capsule,*sample;
        if(!PyArg_ParseTuple(args,"OO",&capsule,&sample))return nullptr;
        auto* evidence=static_cast<Evidence*>(PyCapsule_GetPointer(capsule,EVIDENCE));if(!evidence)throw Error{};
        auto graph=std::make_unique<Adjacency>();graph->owner=evidence;
        graph->evidence_capsule=capsule;Py_INCREF(capsule);
        graph->neighbors.resize(evidence->nodes.size());graph->keys.resize(evidence->nodes.size(),nullptr);
        const char* names[]={"provider","peer","customer"};
        for(int kind=0;kind<3;kind++){
            O table(PyObject_GetAttrString(sample,names[kind]));
            Py_ssize_t pos=0;PyObject *key,*values;
            while(PyDict_Next(table,&pos,&key,&values)){
                Id id=evidence->find(string(key));
                if(id==MISSING){PyErr_SetString(PyExc_ValueError,"adjacency/evidence mismatch");throw Error{};}
                if(!graph->keys[id]){graph->keys[id]=key;Py_INCREF(key);}
                O iterator(PyObject_GetIter(values));
                while(auto* raw=PyIter_Next(iterator)){
                    O neighbor(raw);Id target=evidence->find(string(neighbor));
                    if(target==MISSING){PyErr_SetString(PyExc_ValueError,"adjacency/evidence mismatch");throw Error{};}
                    if(!graph->keys[target]){graph->keys[target]=neighbor;Py_INCREF(neighbor);}
                    graph->neighbors[id][kind].push_back(target);
                }
                if(PyErr_Occurred())throw Error{};
            }
        }
        O result(PyCapsule_New(graph.get(),ADJACENCY,destroy_adjacency));graph.release();return result.release();
    }catch(Error&){return nullptr;}catch(const std::bad_alloc&){return PyErr_NoMemory();}
}
PyObject* expand_ordered(PyObject*,PyObject* args){
    try{
        PyObject *capsule,*dist_capsule,*tree,*raw;
        if(!PyArg_ParseTuple(args,"OOOO",&capsule,&dist_capsule,&tree,&raw))return nullptr;
        auto* graph=static_cast<Adjacency*>(PyCapsule_GetPointer(capsule,ADJACENCY));if(!graph)throw Error{};
        auto* lengths=static_cast<Distances*>(PyCapsule_GetPointer(dist_capsule,DISTANCES));if(!lengths)throw Error{};
        if(graph->owner!=lengths->owner){PyErr_SetString(PyExc_ValueError,"adjacency/distance identities differ");throw Error{};}
        O active(PySequence_Fast(raw,"active paths must be a sequence")),flags(PyObject_GetAttrString(tree,"path_flag")),output(PyDict_New());
        struct Current{PyObject* tuple;std::vector<Id> ids;};
        std::vector<Current> paths;
        for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(active.p);i++){
            auto* path=PySequence_Fast_GET_ITEM(active.p,i);
            if(!PyTuple_Check(path)||PyTuple_GET_SIZE(path)<1){PyErr_SetString(PyExc_ValueError,"active path must be nonempty tuple");throw Error{};}
            if(string(PyTuple_GET_ITEM(path,0))!=lengths->source){PyErr_SetString(PyExc_ValueError,"path source differs from distances");throw Error{};}
            Current current{path,{}};
            for(Py_ssize_t j=0;j<PyTuple_GET_SIZE(path);j++)current.ids.push_back(graph->owner->find(string(PyTuple_GET_ITEM(path,j))));
            paths.push_back(std::move(current));
        }
        const char* names[]={"shell1","shell2","shell3","core1","core2","core3"};
        O keys(PyTuple_New(6));for(int i=0;i<6;i++){auto* key=PyUnicode_FromString(names[i]);if(!key)throw Error{};PyTuple_SET_ITEM(keys.p,i,key);}
        for(int relation=0;relation<3;relation++){
            O flag(PyLong_FromLong(relation));
            for(const auto& current:paths){
                auto size=current.ids.size();auto parent=current.ids.back();
                if(size>=9||parent==MISSING)continue;
                // Read per relationship, as in upstream, since flags may be overwritten.
                O state(PyObject_GetItem(flags,current.tuple));auto status=integer(state);
                const auto& customers=graph->neighbors[parent][2];
                for(auto neighbor:graph->neighbors[parent][relation]){
                    if(std::find(current.ids.begin(),current.ids.end(),neighbor)!=current.ids.end())continue;
                    auto distance=lengths->values[neighbor];if(distance<0)distance=size+1;
                    if(static_cast<int64_t>(size)+1-distance>2)continue;
                    if(status!=0&&std::find(customers.begin(),customers.end(),neighbor)==customers.end())continue;
                    O right(PyTuple_New(size+1));
                    for(size_t j=0;j<size;j++){auto* node=PyTuple_GET_ITEM(current.tuple,j);Py_INCREF(node);PyTuple_SET_ITEM(right.p,j,node);}
                    auto* key=graph->keys[neighbor];Py_INCREF(key);PyTuple_SET_ITEM(right.p,size,key);
                    checked(PyDict_SetItem(flags,right,flag));
                    int bucket=(graph->owner->nodes[neighbor].core?3:0)+(distance<=3?0:distance<=5?1:2);
                    auto* name=PyTuple_GET_ITEM(keys.p,bucket);auto* group=PyDict_GetItemWithError(output,name);
                    if(!group){if(PyErr_Occurred())throw Error{};O values(PyList_New(0));checked(PyDict_SetItem(output,name,values));group=PyDict_GetItem(output,name);}
                    checked(PyList_Append(group,right));
                }
            }
            if(PyErr_CheckSignals()<0)throw Error{};
        }
        return output.release();
    }catch(Error&){return nullptr;}catch(const std::bad_alloc&){return PyErr_NoMemory();}
}
bool rich(PyObject* a,PyObject* b,int op){int value=PyObject_RichCompareBool(a,b,op);checked(value);return value;}
PyObject* update_paths(PyObject*,PyObject* args){
    try{
        PyObject *tree,*paths,*candidate;
        if(!PyArg_ParseTuple(args,"OOO",&tree,&paths,&candidate))return nullptr;
        O scores(PyObject_GetAttrString(tree,"spread_score")),routes(PyObject_GetAttrString(tree,"AS_path"));
        O threshold(PyObject_GetAttrString(tree,"threshold")),predict(PyObject_GetAttrString(tree,"get_model"));
        // Cache only the immutable threshold comparison, retaining NumPy scalar semantics.
        // Own each key until the layer ends so a replaced score cannot reuse its address.
        O above(PyDict_New());
        auto high=[&](PyObject* value){
            // Pointer identity, not scalar equality: NaN and signed zeros remain distinct.
            O identity(PyLong_FromVoidPtr(value));auto* cached=PyDict_GetItemWithError(above,identity);
            if(cached)return PyTuple_GET_ITEM(cached,1)==Py_True;
            if(PyErr_Occurred())throw Error{};
            bool result=rich(value,threshold,Py_GT);
            O entry(PyTuple_Pack(2,value,result?Py_True:Py_False));checked(PyDict_SetItem(above,identity,entry));return result;
        };
        Py_ssize_t pos=0;PyObject *name,*values;
        while(PyDict_Next(paths,&pos,&name,&values)){
            O raw(PyObject_CallFunctionObjArgs(predict,values,name,nullptr)),probabilities(PySequence_Fast(raw,"scores must be a sequence"));
            if(PySequence_Fast_GET_SIZE(probabilities.p)!=PyList_GET_SIZE(values)){PyErr_SetString(PyExc_ValueError,"score count differs");throw Error{};}
            for(Py_ssize_t i=0;i<PyList_GET_SIZE(values);i++){
                auto* path=PyList_GET_ITEM(values,i);auto* score=PySequence_Fast_GET_ITEM(probabilities.p,i);
                checked(PyDict_SetItem(scores,path,score));
                O existing(PyObject_GetItem(routes,PyTuple_GET_ITEM(path,PyTuple_GET_SIZE(path)-1)));
                if(PyList_GET_SIZE(existing.p)==0){checked(PyList_Append(existing,path));checked(PySet_Add(candidate,path));continue;}
                O prior(PyObject_GetItem(scores,PyList_GET_ITEM(existing.p,0)));
                bool prior_high=high(prior);
                // <= cannot be replaced by !greater: preserve NaN behavior.
                if(!prior_high&&rich(prior,threshold,Py_LE)&&rich(prior,score,Py_LT)){
                    Py_INCREF(path);checked(PyList_SetItem(existing,0,path));checked(PySet_Add(candidate,path));
                }else if(prior_high&&high(score)){checked(PyList_Append(existing,path));checked(PySet_Add(candidate,path));}
            }
        }
        Py_INCREF(candidate);return candidate;
    }catch(Error&){return nullptr;}catch(const std::bad_alloc&){return PyErr_NoMemory();}
}
PyMethodDef pipeline_methods[]={
    {"pack",pack,METH_O,"Pack unchanged feature evidence."},
    {"distances",distances,METH_VARARGS,"Pack unchanged distances."},
    {"features",features,METH_VARARGS,"Construct unchanged feature matrices."},
    {"pack_adjacency",pack_adjacency,METH_VARARGS,"Snapshot adjacency in original order."},
    {"expand",expand_ordered,METH_VARARGS,"Expand candidates through integer tables."},
    {"add_paths",update_paths,METH_VARARGS,"Update routes preserving NumPy comparisons."},
    {nullptr,nullptr,0,nullptr}};
PyModuleDef pipeline_module={PyModuleDef_HEAD_INIT,"_pathradar_pipeline",nullptr,-1,pipeline_methods};
}
PyMODINIT_FUNC PyInit__pathradar_pipeline(){return PyModule_Create(&pipeline_module);}
