from dash import Input, Output, State, dcc


def register_callbacks(app) -> None:
    """
        This function initializes various callbacks for the main app. Each callback consists of an @app.callback part and
        the function for the callback itself. Each interaction with the app triggers a callback with one ore more Outputs or
        Inputs and States.
        General callback overview:
        OUTPUTS are the component ids and their properties which change based on the callback.
        Example: An image gets shown after the upload

        INPUTS are the component ids and their properties which are used to create the outputs.
        Example: The contents (e.g. a list) with the uploaded files

        STATES are dynamic properties which need to be considered in the callback.
        Example: The children, e.g. a list with image names, already exist in the app. Without the state property,
        the old image data would be deleted and replaced with the new uploads.
        The function itself has to consider their parameters based on the order of the inputs and the states. When the
        callback has 2 Inputs and one State, the function must have 3 parameters AND RETURN the number of outputs in the
        order in which the Output properties are sorted.

        :param app: The main Dash Application where the callbacks will be registered
        :return: None
        """


