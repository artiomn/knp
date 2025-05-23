"""
@file sonata.py
@brief Import saving and loading to sonata format.

@kaspersky_support Vartenkov A.
@license Apache 2.0 License.
@copyright © 2024 AO Kaspersky Lab
@date 01.11.2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable = no-name-in-module
from knp.base_framework._knp_python_framework_base_framework import save_network, load_network


__all__ = ['save_network', 'load_network']
