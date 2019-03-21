# Schrodinger's Assignment functions
import math
import cmath
import numpy as np
import scipy.constants as c
import scipy.special as s
from mayavi import mlab
# ------------------------------------------

# Week 3

# deg_to_rad and rad_to_deg


def deg_to_rad(deg):
    rad = (deg / 180) * c.pi
    rad = round(rad, 5)
    return rad


def rad_to_deg(rad):
    deg = (rad / c.pi) * 180
    deg = round(deg, 5)
    return deg

# spherical_to_cartesian and cartesian_to_spherical


def safe_arctan(o, a):
    if a == 0 and o != 0:
        return (np.pi)/2
    elif a == 0 and o ==0:
        return round(0,5)
    else:
        return np.arctan(o/a)


def spherical_to_cartesian(r,theta,phi):
    x = (r*(np.sin(theta)))*(np.cos(phi))
    y = (r*(np.sin(theta)))*(np.sin(phi))
    z = r*np.cos(theta)
    return round(x, 5), round(y, 5), round(z, 5)


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = safe_arctan(y,x)
    return round(r, 5), round(theta, 5), round(phi, 5)

# ------------------------------------------

# Week 4

# Angular Wave functions


def angular_wave_func(m, l, theta, phi):
    """calculates the normalised angular solution for the given values, returns a complex number rounded to 5dp for both real and imaginary parts"""
    # the code doesn't work cos there's an extra negative somewhere. removing partA will pass all the test cases in W4Q1 though

    i = np.complex(0, 1)

    if m < 0:
        partA = 1
    elif m == 0:
        partA = 1
    elif m > 0:
        partA = (-1) ** (m)

    partB = ((2 * l + 1) / (4 * c.pi)) * (np.math.factorial(l - np.abs(m)) / np.math.factorial(l + np.abs(m)))
    partBsqrt = np.sqrt(partB)

    partC = cmath.exp(i) ** (m * phi)

    partD = s.lpmv(np.abs(m), l, np.cos(theta))

    normalisedAngularSolution = partA * partBsqrt * partC * partD

    return round(normalisedAngularSolution, 10)


# Radial Wave functions

a = c.physical_constants['Bohr radius'][0]


def radial_wave_func(n, l, r):
    """calculates the normalised radial solution for the given values, returns a value normalised to a**(-3/2) and rounded to 5dp"""

    partA = ((2 / n) ** 3) * (np.math.factorial(n - l - 1) / (
                2 * n * np.math.factorial(n + l)))  # chem eqn says [(n + 1)!]^3 but somehow that's wrong for dw lol
    partAsqrt = np.sqrt(partA)

    partB = math.exp((-r) / (n * a))

    partC = ((2 * r) / (n * a)) ** l

    partD = s.assoc_laguerre((2 * r) / (n * a), n - l - 1, 2 * l + 1)

    normalisedRadialSolution = partAsqrt * partB * partC * partD

    return round(normalisedRadialSolution, 10)
# ------------------------------------------

# Week 5

# mgrid2d to simulate np.mgrid in 2 dimensions


def mgrid2d(xstart, xend, xpoints, ystart, yend, ypoints):
    x_matrix = []
    y_matrix = []
    xstep = (xend-xstart)/(xpoints-1)
    ystep = (yend-ystart)/(ypoints-1)
    x = xstart
    y = ystart
    while x <= xend+xstep/2:
        x_list = []
        for y in range(ypoints):
            x_list.append(x)
        x_matrix.append(x_list)
        x += xstep
    for x in range(xpoints):
        y_list = []
        y = ystart
        while y < yend + ystep/2:
            y_list.append(y)
            y += ystep
        y_matrix.append(y_list)
    array = [x_matrix, y_matrix]
    return array

# mgrid3d to simulate np.mgrid in 3 dimensions


def mgrid3d(xstart, xend, xpoints,
            ystart, yend, ypoints,
            zstart, zend, zpoints):
    xstep = (xend-xstart)/(xpoints-1)
    ystep = (yend-ystart)/(ypoints-1)
    zstep = (zend-zstart)/(zpoints-1)
    x_array = []
    y_array = []
    z_array = []
    x = xstart
    while x < xend + xstep/2:
        xpoint_array = []
        for y in range(ypoints):
            x_list = []
            for z in range(zpoints):
                x_list.append(x)
            xpoint_array.append(x_list)
        x_array.append(xpoint_array)
        x += xstep
    for x in range(xpoints):
        ypoint_array = []
        y = ystart
        while y < yend + ystep/2:
            y_list = []
            for z in range(zpoints):
                y_list.append(y)
            ypoint_array.append(y_list)
            y += ystep
        y_array.append(ypoint_array)
    for x in range(xpoints):
        zpoint_array = []
        for y in range(ypoints):
            z_list = []
            z = zstart
            while z < zend + zstep/2:
                z_list.append(z)
                z += zstep
            zpoint_array.append(z_list)
        z_array.append(zpoint_array)
    ans_array = [x_array, y_array, z_array]
    return ans_array

# ------------------------------------------

# Week6

# Calculating the square of the magnitude of the real wave function


def mag(c):
    real = np.real(c) # Extracts the real value from c
    imaginary = np.imag(c) # Extracts the complex value from c
    magnitude = np.sqrt(real**2 + imaginary**2)
    return magnitude


def hydrogen_wave_func(n, l, m, roa, Nx, Ny, Nz):
    x, y, z = mgrid3d(-1*roa, roa, Nx, -1*roa, roa, Ny, -1*roa, roa, Nz)
    xx = np.array(x)
    yy = np.array(y)
    zz = np.array(z)
    # Vectorize all the functions so that they can take the xx, yy, and zz arrays directly as input
    # without having to use loops
    angular_vector = np.vectorize(angular_wave_func)
    radial_vector = np.vectorize(radial_wave_func)
    spherical_vector = np.vectorize(cartesian_to_spherical)
    mag_vector = np.vectorize(mag)
    round_vector = np.vectorize(round)
    r, theta, phi = spherical_vector(xx, yy, zz) # Convert the input cartesian coordinates to spherical coordinates
    r = r*a
    if m == 0:
        Y = angular_vector(m, l, theta, phi)
    elif m < 0:
        Y = (1j/np.sqrt(2))*(angular_vector(m, l, theta, phi) - (-1**m)*angular_vector(-1*m, l, theta, phi))
    elif m > 0:
        Y = (1j/np.sqrt(2))*(angular_vector(-1*m, l, theta, phi) + (-1**m) * angular_vector(m, l, theta, phi))
    mag_Y = mag_vector(Y) # Extract the magnitude from the complex output of Y
    R = radial_vector(n, l, r)
    magnitude = round_vector((R*mag_Y)**2, 5)
    return [xx, yy, zz, magnitude]

# ------------------------------------------

# Week 9

# Plotting orbitals in Mayavi


# Code to save the data to a file so that
# you don't have to keep on computing it:

n = int(input('Principal Quantum Number (n): '))
l = int(input('Angular Quantum Number (l): '))
m = int(input('Magnetic Quantum Number (m): '))
roa = 50
Nx = int(input('Number of points on the x axis: '))
Ny = int(input('Number of points on the y axis: '))
Nz = int(input('Number of points on the z axis: '))
x,y,z,mag=hydrogen_wave_func(n, l, m, roa, Nx, Nz, Ny)
x.dump('x_test.dat')
y.dump('y_test.dat')
z.dump('z_test.dat')
mag.dump('den_test.dat')

mu, sigma = 0, 0.1
x = np.load('x_test.dat')
y = np.load('y_test.dat')
z = np.load('z_test.dat')

density = np.load('den_test.dat')
figure = mlab.figure('DensityPlot')
pts = mlab.contour3d(density,contours=40,opacity=0.5)
mlab.axes()
mlab.show()


from traits.api import HasTraits, Instance, Array, \
    on_trait_change
from traitsui.api import View, Item, HGroup, Group

from tvtk.api import tvtk
from tvtk.pyface.scene import Scene

from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor, MayaviScene, \
                                MlabSceneModel

################################################################################
# Create some data
data = np.load('den_test.dat')

################################################################################
# The object implementing the dialog
class VolumeSlicer(HasTraits):
    # The data to plot
    data = Array()

    # The 4 views displayed
    scene3d = Instance(MlabSceneModel, ())
    scene_x = Instance(MlabSceneModel, ())
    scene_y = Instance(MlabSceneModel, ())
    scene_z = Instance(MlabSceneModel, ())

    # The data source
    data_src3d = Instance(Source)

    # The image plane widgets of the 3D scene
    ipw_3d_x = Instance(PipelineBase)
    ipw_3d_y = Instance(PipelineBase)
    ipw_3d_z = Instance(PipelineBase)

    _axis_names = dict(x=0, y=1, z=2)


    #---------------------------------------------------------------------------
    def __init__(self, **traits):
        super(VolumeSlicer, self).__init__(**traits)
        # Force the creation of the image_plane_widgets:
        self.ipw_3d_x
        self.ipw_3d_y
        self.ipw_3d_z


    #---------------------------------------------------------------------------
    # Default values
    #---------------------------------------------------------------------------
    def _data_src3d_default(self):
        return mlab.pipeline.scalar_field(self.data,
                            figure=self.scene3d.mayavi_scene)

    def make_ipw_3d(self, axis_name):
        ipw = mlab.pipeline.image_plane_widget(self.data_src3d,
                        figure=self.scene3d.mayavi_scene,
                        plane_orientation='%s_axes' % axis_name)
        return ipw

    def _ipw_3d_x_default(self):
        return self.make_ipw_3d('x')

    def _ipw_3d_y_default(self):
        return self.make_ipw_3d('y')

    def _ipw_3d_z_default(self):
        return self.make_ipw_3d('z')


    #---------------------------------------------------------------------------
    # Scene activation callbaks
    #---------------------------------------------------------------------------
    @on_trait_change('scene3d.activated')
    def display_scene3d(self):
        outline = mlab.pipeline.outline(self.data_src3d,
                        figure=self.scene3d.mayavi_scene,
                        )
        self.scene3d.mlab.view(40, 50)
        # Interaction properties can only be changed after the scene
        # has been created, and thus the interactor exists
        for ipw in (self.ipw_3d_x, self.ipw_3d_y, self.ipw_3d_z):
            # Turn the interaction off
            ipw.ipw.interaction = 0
        self.scene3d.scene.background = (0, 0, 0)
        # Keep the view always pointing up
        self.scene3d.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()


    def make_side_view(self, axis_name):
        scene = getattr(self, 'scene_%s' % axis_name)

        # To avoid copying the data, we take a reference to the
        # raw VTK dataset, and pass it on to mlab. Mlab will create
        # a Mayavi source from the VTK without copying it.
        # We have to specify the figure so that the data gets
        # added on the figure we are interested in.
        outline = mlab.pipeline.outline(
                            self.data_src3d.mlab_source.dataset,
                            figure=scene.mayavi_scene,
                            )
        ipw = mlab.pipeline.image_plane_widget(
                            outline,
                            plane_orientation='%s_axes' % axis_name)
        setattr(self, 'ipw_%s' % axis_name, ipw)

        # Synchronize positions between the corresponding image plane
        # widgets on different views.
        ipw.ipw.sync_trait('slice_position',
                            getattr(self, 'ipw_3d_%s'% axis_name).ipw)

        # Make left-clicking create a crosshair
        ipw.ipw.left_button_action = 0
        # Add a callback on the image plane widget interaction to
        # move the others
        def move_view(obj, evt):
            position = obj.GetCurrentCursorPosition()
            for other_axis, axis_number in self._axis_names.items():
                if other_axis == axis_name:
                    continue
                ipw3d = getattr(self, 'ipw_3d_%s' % other_axis)
                ipw3d.ipw.slice_position = position[axis_number]

        ipw.ipw.add_observer('InteractionEvent', move_view)
        ipw.ipw.add_observer('StartInteractionEvent', move_view)

        # Center the image plane widget
        ipw.ipw.slice_position = 0.5*self.data.shape[
                    self._axis_names[axis_name]]

        # Position the view for the scene
        views = dict(x=( 0, 90),
                     y=(90, 90),
                     z=( 0,  0),
                     )
        scene.mlab.view(*views[axis_name])
        # 2D interaction: only pan and zoom
        scene.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleImage()
        scene.scene.background = (0, 0, 0)


    @on_trait_change('scene_x.activated')
    def display_scene_x(self):
        return self.make_side_view('x')

    @on_trait_change('scene_y.activated')
    def display_scene_y(self):
        return self.make_side_view('y')

    @on_trait_change('scene_z.activated')
    def display_scene_z(self):
        return self.make_side_view('z')


    #---------------------------------------------------------------------------
    # The layout of the dialog created
    #---------------------------------------------------------------------------
    view = View(HGroup(
                  Group(
                       Item('scene_y',
                            editor=SceneEditor(scene_class=Scene),
                            height=250, width=300),
                       Item('scene_z',
                            editor=SceneEditor(scene_class=Scene),
                            height=250, width=300),
                       show_labels=False,
                  ),
                  Group(
                       Item('scene_x',
                            editor=SceneEditor(scene_class=Scene),
                            height=250, width=300),
                       Item('scene3d',
                            editor=SceneEditor(scene_class=MayaviScene),
                            height=250, width=300),
                       show_labels=False,
                  ),
                ),
                resizable=True,
                title='Volume Slicer',
                )


m = VolumeSlicer(data=data)
m.configure_traits()