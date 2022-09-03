#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#include <iostream>
#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/integral_image_normal.h>

namespace py = pybind11;

py::array_t<float> compute_normal(py::array_t<float> points)
{
    py::buffer_info points_buff = points.request();
    int width = points_buff.shape[1];
    int height = points_buff.shape[0];

    if (points_buff.ndim != 3)
        throw std::runtime_error("Number of dimensions must be 2");
    if (points_buff.shape[2] != 3)
        throw std::runtime_error("Shape in the last dimension must be 3");

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    cloud->width = width;
    cloud->height = height;
    cloud->points.resize(cloud->width * cloud->height);
    auto point_array = points.unchecked<3>();

    for (int ri = 0; ri < cloud->height; ++ri)
    {
        for (int ci = 0; ci < cloud->width; ++ci)
        {
            cloud->operator()(ci, ri).x = point_array(ri, ci, 0);
            cloud->operator()(ci, ri).y = point_array(ri, ci, 1);
            cloud->operator()(ci, ri).z = point_array(ri, ci, 2);
        }
    }

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(0.02f);
    ne.setNormalSmoothingSize(10.0f);
    ne.setInputCloud(cloud);
    ne.compute(*normals);

    std::vector<float> data(640 * 480 * 3);
    for (int ri = 0; ri < normals->height; ++ri)
        for (int ci = 0; ci < normals->width; ++ci)
            for (int ch = 0; ch < 3; ++ch)
                data[(3 * ri * 640) + (3 * ci) + ch] = normals->operator()(ci, ri).data_n[ch];

    return py::array_t<float>({height, width, 3}, &data[0]);
}

PYBIND11_MODULE(smooth_normal, m)
{
    m.def("compute_normal", &compute_normal);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}