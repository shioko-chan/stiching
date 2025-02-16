#include <opencv2/opencv.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/array.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <array>
#include <CGAL/disable_warnings.h>

using namespace cv;
using namespace std;

typedef std::array<std::size_t,3> Facet;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3  Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;

struct Construct{
  Mesh& mesh;
  template < typename PointIterator>
  Construct(Mesh& mesh,PointIterator b, PointIterator e) : mesh(mesh){
    for(; b!=e; ++b){
      boost::graph_traits<Mesh>::vertex_descriptor v;
      v = add_vertex(mesh);
      mesh.point(v) = *b;
    }
  }
  Construct& operator=(const Facet f){
    typedef boost::graph_traits<Mesh>::vertex_descriptor vertex_descriptor;
    typedef boost::graph_traits<Mesh>::vertices_size_type size_type;
    mesh.add_face(vertex_descriptor(static_cast<size_type>(f[0])),
                  vertex_descriptor(static_cast<size_type>(f[1])),
                  vertex_descriptor(static_cast<size_type>(f[2])));
    return *this;
  }
  Construct&
  operator*() { return *this; }
  Construct&
  operator++() { return *this; }
  Construct
  operator++(int) { return *this; }
};

class Triangulation{
private:
    vector<Point_3> points;

public:
    void mat2Points(Mat& depMap, int samplingStep);
    void triangulation_3(Mat& depMap);

};

void Triangulation::mat2Points(Mat& depMap, int samplingStep){
    this->points.reserve((depMap.rows/samplingStep)*(depMap.cols/samplingStep));

    for(int i = 0; i < depMap.rows; i += samplingStep){
        for(int j = 0; j < depMap.cols; j += samplingStep){
            uchar depth_value = depMap.at<uchar>(i, j);
            this->points.push_back(Point_3(j, i, depth_value));
        }
    }
}

void Triangulation::triangulation_3(Mat& depMap){
    mat2Points(depMap, 1);

    Mesh mesh;

    Construct construct(mesh, this->points.begin(), this->points.end());
    CGAL::advancing_front_surface_reconstruction(this->points.begin(), this->points.end(), construct);

    ofstream file("../output.off");
    file << mesh;
    file.close();
}