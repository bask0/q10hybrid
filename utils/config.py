
import optuna

from typing import Dict, Union, Any, Iterable, Optional
import os
import re
import shutil
from copy import deepcopy
import warnings


os.environ['WANDB_SILENT'] = 'true'


class TrialNameNotSetError(Exception):
    """Raised if trial name is not set (`--`).

    Attributes:
        message (srt): explanation of the error
    """

    def __init__(self, message=None):
        if message is None:
            message = (
                'the attribute `TRIAL_NAME` is set to its default (`--`). '
                'To use the full functionality of the configuration class, '
                'pass a trial (`str` or `optuna.Trial`) in initialization or '
                'use MyConfig.set_trial_name(...).'
            )
        self.message = message
        super().__init__(self.message)


class MetaAttributesCheck(type):
    def __call__(cls, *args, **kwargs):
        """Checks attributes after initialization."""
        obj = type.__call__(cls, *args, **kwargs)
        obj._test_required_attributes()
        return obj


class BaseConfig(metaclass=MetaAttributesCheck):
    """Defines meta-cofiguration of model. Needs to be subclassed.

    Note:
        * This class needs to be subclassed.
        * Defines a set of global attributes that do not change between model
          tuning, model training or inference. Attributes can change though,
          if they are an agrument to `__init__`.

        Subclass requirements:
        * Subclass `__init__` must include (see `Example`):
          `super(MyConfig, self).__self__(*args, **kwargs)`
        * Set at least these attributes in the subclass `__init__`:
          `STUDY_NAME`, `ROOT_DIR`, `MAX_EPOCHS`.
        * Subclass must take argument `trial`, either s `str`, or an
          `optuna.Tria`.
        * To change name generation from `optuna.Trial`, override the method
          `_name_from_trial`, keep signature `_name_from_trial(self, trial)`.
        * If an attribute needs to be set dymanically, just make it an
          argument of the `__init__` function of the subclass.
        * Attributes to record need to be all upper case, `_` allowed but not
          at beginning. For example `MYATTR` nad `MY_ATTR` get recorded, but
          `_MY_ATTR` or `MyAttr` not.

    Example:
        Standard usage: create subclass and set configuration. Some attributes
        are required, but add global attributes as you wish.

        >>> from utils import ModelConfig
        ... class MyConfig(ModelConfig):
        ...     def __init__(self, batch_size, *args, **kwargs):
        ...         # Required:
        ...         super(MyConfig, self).__self__(*args, **kwargs)
        ...
        ...         # Required:
        ...         self.STUDY_NAME = 'awesome_study'
        ...         self.ROOT_DIR = 'my/dir/to'
        ...         self.MIN_EPOCHS = 5
        ...         self.MAX_EPOCHS = 100
        ...
        ...         # Custom global atrributes:
        ...         self.BATCH_SIZE = batch_size
        ...         self.MY_ATTR = 'my_value'
        >>> config = MyConfig(trial='test', batch_size=32)
        >>> config.BATCH_SIZE
        32
        >>> config.STUDY_DIR
        'my/dir/to/awesome_study'

    Attributes:
        Required:
        ROOT_DIR (str): the base project directory. Defaults to `logs`.
        STUDY_NAME (str): the study name. No default.
        MIN_EPOCHS (int): the minimum epochs to run before pruning. Defaults to 2.
        MAX_EPOCHS (int): the maximum number of epochs to run. No default.


        Generated:
        STUDY_DIR (str): the study directory [auto generated].
        TRIAL_NAME (str): the trial name [auto generated].
        TRIAL_DIR (str): the model directory [auto generated].

    Args:
        trial (Union[str, optuna.Trial]): either a trial name of an
            optuna trial, from which a name will be derived.
        ... (Any): your custom, non-hard-coded configuration.

    """

    _VERSION = None
    REQUIRED_ATTRS = ['STUDY_NAME', 'ROOT_DIR', 'MIN_EPOCHS', 'MAX_EPOCHS']
    DERIVED_ATTRS = ['STUDY_DIR', 'TRIAL_NAME', 'TRIAL_DIR', 'OPTUNA_DB']

    def __init__(
            self,
            trial: Union[str, optuna.Trial] = '--',
            resume_policy: Optional[Union[str, bool]] = False,
            reset_version: bool = False) -> None:
        """Study meta-cofiguration. Needs to be subclassed.

        Args:
            trial (Optional[Union[str, optuna.Trial]]): A trial name of an optuna trial. Default is '--', which means
                the `TRIAL_NAME` and derived attributes cannot be used.
            resume_policy (Optional[Union[str, bool]]): Whether to resume or create a new verion directory. Defaults
                to `False`. Options:
                * `False`: a new version is created.
                * `True`: use latest version (create new one if none exists).
                * `'latest'`: use latest version (fail if none exists).
                * `'version_[int]'`: use version `[int]` (fail if not existig). [int] must be castable to integer.
            resume_policy (bool): if `True`, the current version will be reset. A version is infered from
                `resume_policy`. This can have unintended side-effects. If `False`, the version will be infered when
                initialized first and all consecutive initializations during runtime will use the same version.
                Defaults to `False`.
        """

        if reset_version:
            BaseConfig._VERSION = None

        if BaseConfig._VERSION is not None:
            warnings.warn(
                f'a configuration has already been initialized. The version `{self._VERSION}` from '
                'the first call will be used.'
            )

        # Don't set manually after class initialization, breaks stuff.
        self._resume_policy = self.__check_resume_policy(resume_policy)

        self.TRIAL_NAME = self.__get_trial_name(trial)
        self.ROOT_DIR = 'logs/'

        self.MIN_EPOCHS = 2

        self.is_resumed = False

    def set_trial_name(self, trial):
        """Manually set the trial name after initialization.

        Args:
            trial (Union[str, optuna.Trial]): A trial name of an optuna trial.
        """
        self.TRIAL_NAME = self.__get_trial_name(trial)

    @property
    def STUDY_NAME(self) -> str:
        return self._STUDY_NAME

    @STUDY_NAME.setter
    def STUDY_NAME(self, x) -> None:
        self.__maybe_raise('STUDY_NAME', x, str)
        self._STUDY_NAME = x

    @property
    def TRIAL_NAME(self) -> str:
        if self._TRIAL_NAME == '--':
            raise TrialNameNotSetError()
        return self._TRIAL_NAME

    @TRIAL_NAME.setter
    def TRIAL_NAME(self, x) -> None:
        self.__maybe_raise('TRIAL_NAME', x, str)
        self._TRIAL_NAME = x

    @property
    def TRIAL_UID(self) -> str:
        return '{}_v{}'.format(self.TRIAL_NAME, self.VERSION_NR)

    @property
    def ROOT_DIR(self) -> str:
        return self._ROOT_DIR

    @ROOT_DIR.setter
    def ROOT_DIR(self, x) -> None:
        self.__maybe_raise('ROOT_DIR', x, str)
        self._ROOT_DIR = x

    @property
    def MIN_EPOCHS(self) -> int:
        return self._MIN_EPOCHS

    @MIN_EPOCHS.setter
    def MIN_EPOCHS(self, x) -> None:
        self.__maybe_raise('MIN_EPOCHS', x, int)
        self._MIN_EPOCHS = x

    @property
    def MAX_EPOCHS(self) -> int:
        return self._MAX_EPOCHS

    @MAX_EPOCHS.setter
    def MAX_EPOCHS(self, x) -> None:
        self.__maybe_raise('MAX_EPOCHS', x, int)
        self._MAX_EPOCHS = x

    @property
    def STUDY_DIR(self) -> str:
        return os.path.join(self.ROOT_DIR, self.STUDY_NAME)

    @property
    def STUDY_DIR_VERSION(self) -> str:
        if BaseConfig.VERSION is None:
            raise AssertionError(
                'cannot access `STUDY_DIR_VERSION` as not `VERSION` has been set.'
            )
        return os.path.join(self.ROOT_DIR, self.STUDY_NAME, str(self.VERSION))

    @property
    def OPTUNA_DB(self) -> str:
        os.makedirs(self.STUDY_DIR_VERSION, exist_ok=True)
        p = os.path.join(self.STUDY_DIR_VERSION, 'optuna_runs.db')
        return f'{"sqlite:///" + p}'

    @property
    def TRIAL_DIR(self) -> str:
        return os.path.join(self.STUDY_DIR_VERSION, self.TRIAL_UID)

    @property
    def LAST_CKPT_PATH(self) -> str:
        return os.path.join(self.STUDY_DIR_VERSION, 'last.ckpt')

    @property
    def resume_policy(self) -> Union[str, bool]:
        # resume_policy cannot be set manually.
        return self._resume_policy

    def __check_resume_policy(self, x) -> Union[str, bool]:
        self.__maybe_raise('resume_policy', x, [str, bool])
        if isinstance(x, str):
            if not (x == 'latest' or x[:8] == 'version_'):
                raise ValueError(
                    f"attempt to set attribute `resume_policy` with value `'{x}'` but expected either "
                    "`'latest'` or a string starting with `'version_'`."
                )
        return x

    def makedirs(self) -> None:
        """Autogenerate required directories, no error when existing.
        """
        os.makedirs(self.TRIAL_DIR, exist_ok=True)

    def get_latest_checkpoint(self, missing_ok=True) -> Union[None, str]:
        """Get the latest model checkpoint.

        Note:
            The checkpoint is expected to have the signature [self.TRIAL_DIR]/last.ckpt

        Args:
            missing_ok (bool, optional): wheter to ignore missing checkpoint file or raise an error.
                Defaults to True.

        Returns:
            Union[None, str]: Either a path (str) or `None` if no checkpoint is available.
        """

        if os.path.isfile(self.LAST_CKPT_PATH):
            return self.LAST_CKPT_PATH
        else:
            if not missing_ok:
                raise FileNotFoundError(
                    f'checkpoint file `{self.LAST_CKPT_PATH}` does not exist. Call '
                    '`MyConfig.resume_path(not_exist_ok=True)` to ignore missing checkpoint file.'
                )
            return None

    def __id2version(self, i: int) -> str:
        return f'version_{i}'

    def __version2id(self, version: str) -> int:
        m = re.match(r'version_(\d+)', version)
        if m is not None:
            m = int(m.group(1))
        return m

    @property
    def VERSION_NR(self) -> int:
        """Returns the version number.

        Returns:
            int: the version number.
        """
        return self.__version2id(self.VERSION)

    @property
    def VERSION(self) -> str:
        """Returns the current version.

        Note:
            The version is set (and infered) when first called. Further calls access the attribute set before. The
            behavior debends on the attribute `resume_policy`:
            * `False`: a new version is created.
            * `True`: use latest version (create new one if none exists).
            * `'latest'`: use latest version (fail if none exists).
            * `'version_xx'`; use version `xx` (fail if not existig).

        Returns:
            str: the version of signature `'version_X' where XX is the version number.`
        """
        def message(value):
            s = (
                'attempt to infer the version'
            )
            return s

        if BaseConfig._VERSION is not None:
            return BaseConfig._VERSION

        # Infer version.
        latest_version = None  # Is set to newest version if existing.
        current_version = None  # Is set to newest version + 1 if existing, else to `0`.

        existing_versions = []  # Holds existing versions as integers. Empty means no existing versions.
        if os.path.isdir(self.STUDY_DIR):
            for d in os.listdir(self.STUDY_DIR):
                v = self.__version2id(d)
                if v is not None:
                    existing_versions.append(v)
            if len(existing_versions) == 0:
                current_version = 0  # Current version is `0`, no latest.
            else:
                latest_version = max(existing_versions)  # Latest version is set.
                current_version = max(existing_versions) + 1  # Current version is latest +1.
        else:
            current_version = 0  # Current version is `0`, no latest.

        if self.resume_policy is False:
            # If no resuming, take current.
            BaseConfig._VERSION = self.__id2version(current_version)

        elif self.resume_policy is True:
            # If resuming, take latest if present, else current.
            if latest_version is None:
                BaseConfig._VERSION = self.__id2version(current_version)
            else:
                BaseConfig._VERSION = self.__id2version(latest_version)
                self.is_resumed = True

        elif self.resume_policy == 'latest':
            # If resuming `latest`, take latest if present, else raise error.
            if latest_version is None:
                raise AssertionError(
                    'attempt to resume from latest version failed as no versions exist. To create a new version if '
                    'no version exists, set `resume_policy=True` on class initlaization.'
                )
            else:
                BaseConfig._VERSION = self.__id2version(latest_version)
                self.is_resumed = True

        elif self.resume_policy[:8] == 'version_':
            # If resuming using specific version, check if 1) is valid version signature, 2) exists.
            requested_version = self.__version2id(self.resume_policy)
            if requested_version is None:
                raise AssertionError(
                    f'attempt to resume version from `{self.resume_policy}` failed: '
                    'not a valid signature (`version_[int]`).'
                )
            if requested_version not in existing_versions:
                raise AssertionError(
                    f'attempt to resume version from `{self.resume_policy}` failed: version does not exist.'
                )

            BaseConfig._VERSION = self.__id2version(requested_version)
            self.is_resumed = True

        else:
            # Should not be reachable as options handeled before.
            raise AssertionError(
                f'attempt to get the current version failed: the case `requested_version={requested_version}` '
                'is not handeled.'
            )

        if self.is_resumed:
            if not os.path.isfile(self.OPTUNA_DB.split('///')[1]):
                raise AssertionError(
                    f'attempt to resume run from dir `{self.STUDY_DIR_VERSION}` failed. The optuna study database '
                    f'`{self.OPTUNA_DB}` is missing. Please delete the version manually or chose a different '
                    '`resume_policy` policy.'
                )

    def __get_trial_name(self, trial) -> str:
        """Set the trial name.

        Args:
            trial (str or optuna.Trial): A trial name of an optuna trial.

        Returns:
            str: the trial name.

        Raises:
            TypeError: raised if type requirements are not met.

        """
        if isinstance(trial, str):
            return trial
        elif isinstance(trial, optuna.Trial):
            return f'trial_{trial.number}'
        else:
            raise TypeError(
                'attempt to derive `TRIAL_NAME` from `trial`. Argument '
                '`trial` must be of type `str` of `optune.trial.Trial` '
                f'but is of type `{type(trial).__name__}`.'
            )

    def get_params(self, detailed: bool = False) -> Dict:
        """Returns all parameters ni upper case not starting with a `_`.

        Note: this method returns a dict of study parameters. Only attributes
        (and properties if `detailed=True`) in all upper case, `_` allowed but
        not at beginning. For example `MYATTR` nad `MY_ATTR` gets recorded, but
        `_MY_ATTR` or `MyAttr` not.

        Args:
            detailed (bool): if `True`, also class properties are added. Else, only user Attributes (not properties)
                are recorded. Defaults to `False`.

        Returns:
            Dict: a dictionary of parameters.
        """
        params = dict()
        attrs = dir(self) if detailed else self.__dict__
        missing_trial_name = False
        for attr in attrs:
            if all([a.isupper() for a in attr if a != '_']) \
                    and attr[:1] != '_':
                try:
                    params.update({attr: self.__getattribute__(attr)})
                except TrialNameNotSetError as e:
                    params.update({attr: '<TRIAL_NAME>*'})
                    missing_trial_name = True

        if missing_trial_name:
            params.update({'*': 'TRIAL_NAME not set'})

        if detailed:
            params.pop('DERIVED_ATTRS')
            params.pop('REQUIRED_ATTRS')

        return params

    def set_trial_attrs(self, trial: optuna.Trial) -> None:
        """Adds configuration to a trial's user attributes.

        Args:
            trial (optuna.Trial): an optuna Trial.
        """
        for key, value in self.get_params(detailed=True).items():
            trial.set_user_attr(key, value)

    def __str__(self) -> str:
        s = type(self).__name__ + '\n----------'

        p = self.get_params(detailed=True)

        max_k = 0
        max_v = 0
        for k, v in p.items():
            try:
                v = str(v)
                if len(v) > 30:
                    v = v[:10] + '...' + v[-17:]
            except Exception as e:
                v = '<object>'

            if len(k) > max_k:
                max_k = len(k)

            if len(v) > max_v:
                max_v = len(v)

            p[k] = v

        n_max = max_k + max_v + 1

        for k, v in p.items():
            n_fill = n_max - len(k) - len(v)
            s += f'\n{k}: {"." * n_fill} {v}'

        return s

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self):
        """Return a deepcopy of self.

        Returns:
            BaseClass: a copy of the instance.
        """
        return deepcopy(self)

    def __maybe_raise(
            self,
            attr_name: str,
            value: Any,
            expected_type: Union[Iterable[type], type],
            allowed_options: Optional[Iterable[Any]] = None) -> None:
        """Check value type and raise TypeError if mismatch.

        Args:
            attr_name (str): name of the attribute to be set.
            value (Any): the candidate value.
            expected_type (Union[Iterable[type], type]): the required type (or an iterable of such).
            allowed_options (Optional[Iterable[Any]]): if passed, the `value` must be one of the options. Must be an
                iterable (not str).

        Raises:
            TypeError: raised if type requirements are not met.
            ValueError: raised if value is not one of the `allowed_options`.
        """

        if isinstance(expected_type, type):
            expected_type = [expected_type]

        is_of_type = False
        for t in expected_type:
            if isinstance(value, t):
                is_of_type = True

        if not is_of_type:
            value_type = type(value).__name__
            expected_type = ' | '.join([f'`{t.__name__}`' for t in expected_type])
            raise TypeError(
                f'attempt to set `{attr_name}` with a value of type '
                f'`{value_type}` but expected one of: {expected_type}.'
            )

        if allowed_options is not None:
            if isinstance(allowed_options, str):
                raise TypeError(
                    'the argument `allowed_options` cannot be of type `str` but any other iterable.'
                )
            elif not hasattr(allowed_options, '__iter__'):
                raise TypeError(
                    'the argument `allowed_options` must be an iterable (but not a strong).'
                )

            if value not in allowed_options:
                allowed_options = ' | '.join(
                    [f"`'{o}'`" if isinstance(o, str) else f"`{o}`" for o in allowed_options]
                )
                raise ValueError(
                    f'attempt to set `{attr_name}` with a value `{value}` but expected one of: {allowed_options}.'
                )

    def _test_required_attributes(self):
        """Test required attributes.

        Raises:
            AttributeError: Error listing required but missing attributes.
        """

        not_found = []
        for attr in self.REQUIRED_ATTRS:
            if not hasattr(self, attr):
                not_found.append(attr)

        if len(not_found) > 0:
            s = 'the required attribute(s) no available in study config:'
            for attr in not_found:
                s += f' `{attr}`'
            s += '.'

            raise AttributeError(s)

        attr_errors = [
            'some of the automatically generated attributes could not be '
            'derived. Make sure that all deriveable attributes can be '
            'accessed. The following AttributeErrors were raised:'
        ]

        for at in self.DERIVED_ATTRS:
            try:
                getattr(self, at)
            except AttributeError as e:
                attr_errors.append(f'{at}: {e.args[0]}')
            except TrialNameNotSetError:
                pass

        if len(attr_errors) > 1:
            raise AttributeError('\n'.join(attr_errors))

    def self_destruct_version(self):
        """Deletes the entire run.
        """

        shutil.rmtree(self.STUDY_DIR_VERSION)

    def self_destruct_study(self):
        """Deletes all present runs.s
        """

        shutil.rmtree(self.ROOT_DIR)