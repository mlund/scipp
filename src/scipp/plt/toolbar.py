# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

import ipywidgets as ipw

from .. import config
# from .resampling_model import ResamplingMode


def set_button_color(button, selected=False):
    name = 'button_selected' if selected else 'button'
    try:
        button.style.button_color = config['colors'][name]
    except KeyError:
        pass  # we do not have a color we can use


def _make_toggle_button(**kwargs):
    button = ipw.ToggleButton(layout={
        "width": "34px",
        "padding": "0px 0px 0px 0px"
    },
                              **kwargs)
    # set_button_color(button)
    return button


class PlotToolbar:
    """
    Custom toolbar with additional buttons for controlling log scales and
    normalization, and with back/forward buttons removed.
    """
    def __init__(self, external_toolbar=None):
        self._dims = None
        self.controller = None

        self.container = ipw.VBox()
        self.members = {}

        # Keep a reference to the matplotlib toolbar so we can call the zoom
        # and pan methods
        self.external_toolbar = external_toolbar

        self.add_button(name="home_view", icon="home", tooltip="Reset original view")
        # self._resampling_mode = _make_toggle_button(
        #     description='',
        #     tooltip="Switch current resampling mode. Options are 'sum' and 'mean'."
        #     " The correct mode depends on the interpretation of data."
        #     " The default value is guessed based on the data unit but in practice"
        #     " the automatic selection cannot always be relied on.")

    # def initialize(self, log_axis_buttons, button_states):
    #     # self.members['resampling_mode'] = self._resampling_mode
    #     # if 'resampling_mode' in button_states:
    #     #     resampling_modes = {
    #     #         ResamplingMode.mean: (True, 'mean'),
    #     #         ResamplingMode.sum: (False, 'sum')
    #     #     }
    #     #     mode = button_states.pop('resampling_mode')
    #     #     state, description = resampling_modes[mode]
    #     #     self._resampling_mode.description = description
    #     #     self.toggle_button_color(self._resampling_mode, value=state)
    #     self._log_axis = {
    #         dim: _make_toggle_button(tooltip=f'log({dim})')
    #         for dim in log_axis_buttons
    #     }
    #     for name, state in button_states.items():
    #         button = self._log_axis[name[4:]] if name.startswith(
    #             'log_') else self.members[name]
    #         self.toggle_button_color(button, value=state)

    # @property
    # def dims(self):
    #     return self._dims

    # @dims.setter
    # def dims(self, dims):
    #     if self._dims == dims:
    #         return
    #     self._dims = dims
    #     for dim, button in self._log_axis.items():
    #         if dim in self.dims:
    #             button.layout.display = ''
    #         else:
    #             button.layout.display = 'none'
    #     for ax, dim in zip('xyz', dims[::-1]):  # might have only 1 dim -> x
    #         self._log_axis[dim].description = f'log{ax}'
    #         self.members[f'toggle_{ax}axis_scale'] = self._log_axis[dim]
    #     self._update_container()

    def _ipython_display_(self):
        """
        IPython display representation for Jupyter notebooks.
        """
        return self._to_widget()._ipython_display_()

    def _to_widget(self):
        """
        Return the VBox container
        """
        return self.container

    def set_visible(self, visible):
        """
        Need to hide/show the toolbar when a canvas is hidden/shown.
        """
        self.container.layout.display = None if visible else 'none'

    def add_button(self, name, **kwargs):
        """
        Create a new button and add it to the toolbar members list.
        """
        button = ipw.Button(**self._parse_button_args(**kwargs))
        set_button_color(button)
        self.members[name] = button

    def add_togglebutton(self, name, value=False, **kwargs):
        """
        Create a fake ToggleButton using Button because sometimes we want to
        change the value of the button without triggering an update, e.g. when
        we swap the axes.
        """
        button = ipw.ToggleButton(layout={
            "width": "34px",
            "padding": "0px 0px 0px 0px"
        },
                                  value=value,
                                  **kwargs)
        # button = ipw.Button(**self._parse_button_args(**kwargs))
        # set_button_color(button)
        # setattr(button, "value", value)
        # self.toggle_button_color(owner=button, value=value)
        # # Add a local observer to change the color of the button according to
        # # its value.
        # button.on_click(self.toggle_button_color)
        self.members[name] = button

    # def toggle_button_color(self, owner, value=None):
    #     """
    #     Change the color of the button to make it look like a ToggleButton.
    #     """
    #     if value is None:
    #         owner.value = not owner.value
    #     else:
    #         owner.value = value
    #     set_button_color(owner, selected=owner.value)

    def connect(self, view):
        """
        Connect callbacks to button clicks.
        """
        for key, button in self.members.items():
            obj = self if hasattr(self, key) else view
            callback = getattr(obj, key)
            if isinstance(button, ipw.ToggleButton):
                button.observe(callback, names='value')
            else:
                button.on_click(callback)
            # if hasattr(controller, key):
            #     self.members[key].on_click(getattr(controller, key))
            # elif self.members[key] is not None:
            #     if not isinstance(self.members[key], ipw.ToggleButton):
            #         self.members[key].on_click(getattr(self, key))
        # for dim, button in self._log_axis.items():
        #     button.observe(getattr(controller, 'toggle_dim_scale')(dim), 'value')
        # self._resampling_mode.observe(self.toggle_resampling_mode, 'value')
        # if hasattr(controller, 'toggle_resampling_mode'):
        #     self._resampling_mode.observe(controller.toggle_resampling_mode, 'value')

    def _update_container(self):
        """
        Update the container's children according to the buttons in the
        members.
        """
        self.container.children = tuple(self.members.values())

    def _parse_button_args(self, layout=None, **kwargs):
        """
        Parse button arguments and add some default styling options.
        """
        args = {"layout": {"width": "34px", "padding": "0px 0px 0px 0px"}}
        if layout is not None:
            args["layout"].update(layout)
        for key, value in kwargs.items():
            if value is not None:
                args[key] = value
        return args

    @property
    def tool_active(self):
        return self.members["zoom_view"].value or \
                self.members["pan_view"].value

    def home_view(self, button):
        self.external_toolbar.home()

    def pan_view(self, change):
        if change["new"]:
            # In case the zoom button is selected, we need to de-select it
            if self.members["zoom_view"].value:
                self.members["zoom_view"].value = False
                # self.toggle_button_color(self.members["zoom_view"])
            self.external_toolbar.pan()

    def zoom_view(self, change):
        if change["new"]:
            # In case the pan button is selected, we need to de-select it
            if self.members["pan_view"].value:
                self.members["pan_view"].value = False
                # self.toggle_button_color(self.members["pan_view"])
            self.external_toolbar.zoom()

    def save_view(self, button):
        self.external_toolbar.save_figure()

    def rescale_on_zoom(self):
        return self.members["zoom_view"].value

    # def set_resampling_mode_display(self, display):
    #     self._resampling_mode.layout.display = 'block' if display else 'none'

    # def toggle_resampling_mode(self, button):
    #     if self._resampling_mode.value:
    #         self._resampling_mode.description = 'mean'
    #     else:
    #         self._resampling_mode.description = 'sum'


class PlotToolbar1d(PlotToolbar):
    """
    Custom toolbar for 1d figures.
    """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.add_togglebutton(name="pan_view", icon="arrows", tooltip="Pan")
        self.add_togglebutton(name="zoom_view", icon="square-o", tooltip="Zoom")
        self.add_button(name="rescale_to_data", icon="arrows-v", tooltip="Rescale")
        self.members['toggle_xaxis_scale'] = None
        self.add_togglebutton(name="toggle_norm",
                              description="logy",
                              tooltip="log(data)")
        # self.members['resampling_mode'] = None
        self.add_button(name="save_view", icon="save", tooltip="Save")
        self._update_container()


class PlotToolbar2d(PlotToolbar):
    """
    Custom toolbar for 2d figures.
    """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.add_togglebutton(name="pan_view", icon="arrows", tooltip="Pan")
        self.add_togglebutton(name="zoom_view", icon="square-o", tooltip="Zoom")
        self.add_button(name="rescale_to_data", icon="arrows-v", tooltip="Rescale")
        self.add_button(name="transpose", icon="retweet", tooltip="Transpose")
        self.add_togglebutton('toggle_xaxis_scale', description="logx")
        self.add_togglebutton('toggle_yaxis_scale', description="logy")
        self.add_togglebutton(name="toggle_norm",
                              description="log",
                              tooltip="log(data)")
        # self.members['resampling_mode'] = None
        self.add_button(name="save_view", icon="save", tooltip="Save")
        self._update_container()


class PlotToolbar3d(PlotToolbar):
    """
    Custom toolbar for 3d figures.
    """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.add_button(name="camera_x_normal",
                        icon="camera",
                        description="X",
                        tooltip="Camera to X normal. "
                        "Click twice to flip the view direction.")
        self.add_button(name="camera_y_normal",
                        icon="camera",
                        description="Y",
                        tooltip="Camera to Y normal. "
                        "Click twice to flip the view direction.")
        self.add_button(name="camera_z_normal",
                        icon="camera",
                        description="Z",
                        tooltip="Camera to Z normal. "
                        "Click twice to flip the view direction.")
        self.add_togglebutton(name="toggle_axes_helper",
                              value=True,
                              description="\u27C0",
                              style={"font_weight": "bold"},
                              tooltip="Toggle visibility of XYZ axes")
        self.add_togglebutton(name="toggle_outline",
                              value=True,
                              icon="codepen",
                              tooltip="Toggle visibility of outline box")
        self.add_button(name="rescale_to_data", icon="arrows-v", tooltip="Rescale")
        self.add_togglebutton(name="toggle_norm",
                              description="log",
                              tooltip="log(data)")
        # self.members['resampling_mode'] = None

    def home_view(self, button):
        self.external_toolbar.reset_camera()

    def camera_x_normal(self, button):
        self.external_toolbar.camera_x_normal()

    def camera_y_normal(self, button):
        self.external_toolbar.camera_y_normal()

    def camera_z_normal(self, button):
        self.external_toolbar.camera_z_normal()

    def toggle_axes_helper(self, button):
        self.external_toolbar.toggle_axes_helper(button.value)

    def toggle_outline(self, button):
        self.external_toolbar.toggle_outline(button.value)
