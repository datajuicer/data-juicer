import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

from loguru import logger

from data_juicer.utils.lazy_loader import LazyLoader

from ...base_op import Mapper
from ...mixins import EventDrivenMixin, NotificationMixin

label_studio_sdk = LazyLoader("label_studio_sdk")

# Common annotation event types
ANNOTATION_EVENTS = {
    "TASK_CREATED": "task_created",
    "BATCH_CREATED": "batch_created",
    "ANNOTATION_COMPLETED": "annotation_completed",
    "BATCH_ANNOTATION_COMPLETED": "batch_annotation_completed",
    "ERROR_OCCURRED": "error_occurred",
}


class BaseAnnotationMapper(EventDrivenMixin, NotificationMixin, Mapper, ABC):
    """Base class for annotation operations with event-driven capabilities.

    This class provides functionality for creating annotation tasks, waiting
    for annotations to be completed, and handling notification events for
    various stages of the annotation process.

    It supports sending notifications with annotation platform URLs included,
    making it easy for annotators to directly access the annotation interface
    when they receive a notification. This is particularly useful for time-se-
    -nsitive annotation tasks or for coordinating with remote annotation teams.

    Notifications can be sent via email, Slack, or DingTalk, and can be
    configured to trigger on various events such as task creation, batch
    creation, annotation completion, and error conditions.

    URL inclusion in notifications can be enabled or disabled using the
    `include_urls_in_notifications` parameter.
    """

    _batched_op = True  # Mark this as a batched operator

    def __init__(
        self,
        project_name_prefix: str = "DataJuicer_Annotation",
        wait_for_annotations: bool = False,
        timeout: int = 3600,
        poll_interval: int = 60,
        samples_per_task: int = 1,
        max_tasks_per_batch: int = 100,
        project_id: Optional[int] = None,
        notification_config: Optional[Dict] = None,
        notification_events: Optional[Dict[str, bool]] = None,
        **kwargs,
    ):
        """Initialize the base annotation operation

        Args:
            project_name_prefix: Prefix for the project name
            project_id: ID of existing project (if None, creates new project)
            wait_for_annotations: Whether to wait for annotations to complete
            timeout: Maximum time to wait for annotations in seconds
            poll_interval: Time between annotation status checks in seconds
            samples_per_task: Number of samples in each annotation task
            max_tasks_per_batch: Maximum number of tasks in a single batch
            notification_config: Configuration for notifications (email, slack)
            notification_events: Events that should trigger notifications
        """
        # Ensure notification_config is passed to kwargs for NotificationMixin
        kwargs["notification_config"] = notification_config or {}

        # Initialize parent classes
        super().__init__(**kwargs)

        # Store configuration
        self.project_name = project_name_prefix + "_" + str(uuid.uuid4())[:6]
        self.project_id = project_id
        self.wait_for_annotations = wait_for_annotations
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.samples_per_task = samples_per_task
        self.max_tasks_per_batch = max_tasks_per_batch

        # Configure which events should trigger notifications
        self.notification_events = notification_events or {
            "task_created": False,
            "batch_created": True,
            "annotation_completed": False,
            "batch_annotation_completed": True,
            "error_occurred": True,
        }

        # Track task IDs and sample mappings
        self.sample_to_task_id = {}  # Maps sample ID to task ID
        self.task_to_samples = defaultdict(list)  # Maps task ID to sample IDs
        self.processed_annotations = set()
        self.batch_counter = 0

        # Set up event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Set up default event handlers"""
        # Register event handlers with notification support
        self.register_event_handler(ANNOTATION_EVENTS["TASK_CREATED"], self._handle_task_created)
        self.register_event_handler(ANNOTATION_EVENTS["BATCH_CREATED"], self._handle_batch_created)
        self.register_event_handler(ANNOTATION_EVENTS["ANNOTATION_COMPLETED"], self._handle_annotation_completed)
        self.register_event_handler(
            ANNOTATION_EVENTS["BATCH_ANNOTATION_COMPLETED"], self._handle_batch_annotation_completed
        )
        self.register_event_handler(ANNOTATION_EVENTS["ERROR_OCCURRED"], self._handle_error)

    def _handle_task_created(self, data):
        """Handle task created event with notification"""
        task_id = data.get("task_id")
        sample_ids = data.get("sample_ids", [])

        logger.debug(f"Task {task_id} created with {len(sample_ids)} samples")

        # Send notification if configured
        if self.notification_events.get("task_created", False):
            self.send_notification(
                f"Task {task_id} created with {len(sample_ids)} "
                f"samples for annotation in project {self.project_name}",
                subject=f"New Annotation Task Created - {self.project_name}",
            )

    def _handle_batch_created(self, data):
        """Handle batch created event with notification"""
        batch_id = data.get("batch_id")
        task_count = data.get("task_count", 0)
        sample_count = data.get("sample_count", 0)

        logger.info(f"Batch {batch_id} created with {task_count} tasks containing " f"{sample_count} samples")

        # Send notification if configured
        if self.notification_events.get("batch_created", False):
            # Build message with or without URL
            message = (
                f"Batch {batch_id} created with {task_count} tasks containing "
                f"{sample_count} samples in project {self.project_name}"
            )

            project_url = self._get_project_url()
            if project_url:
                message += f"\n\nAnnotation URL: {project_url}"

            self.send_notification(message, subject=f"New Annotation Batch Created - {self.project_name}")

    def _handle_annotation_completed(self, data):
        """Handle annotation completed event with notification"""
        task_id = data.get("task_id")
        annotation_id = data.get("annotation_id")

        logger.debug(f"Annotation {annotation_id} completed for task {task_id}")

        # Mark this task as processed
        self.processed_annotations.add(task_id)

        # Send notification if configured
        if self.notification_events.get("annotation_completed", False):
            sample_count = len(self.task_to_samples.get(task_id, []))

            # Build message with or without URL
            message = (
                f"Annotation {annotation_id} completed for task {task_id} "
                f"with {sample_count} samples in project {self.project_name}"
            )

            # Add URL if configured to include it
            task_url = self._get_task_url(task_id)
            if task_url:
                message += f"\n\nTask URL: {task_url}"

            self.send_notification(message, subject=f"Annotation Completed - {self.project_name}")

    def _handle_batch_annotation_completed(self, data):
        """Handle batch annotation completed event with notification"""
        batch_id = data.get("batch_id")
        task_count = data.get("task_count", 0)
        sample_count = data.get("sample_count", 0)

        if self.notification_events.get("batch_annotation_completed", False):
            # Build message with or without URL
            message = (
                f"Batch {batch_id} done with {task_count} tasks containing "
                f"{sample_count} samples in project {self.project_name}"
            )

            project_url = self._get_project_url()
            if project_url:
                message += f"\n\nProject URL: {project_url}"

            self.send_notification(message, subject=f"Batch Annotation Completed - {self.project_name}")

    def _handle_error(self, data):
        """Handle error event with notification"""
        error_message = data.get("message", "Unknown error")
        task_id = data.get("task_id", "Unknown task")

        logger.error(f"Error in annotation task {task_id}: {error_message}")

        # Send notification if configured
        if self.notification_events.get("error_occurred", False):
            self.send_notification(
                f"Error in annotation task {task_id}: {error_message}",
                subject=f"Annotation Error - {self.project_name}",
                notification_type="email",  # Always send errors via email
            )

    @abstractmethod
    def _create_tasks_batch(self, tasks_data: List[Dict], sample_ids: List[Any]) -> List[int]:
        """Create multiple tasks in the annotation platform

        Args:
            tasks_data: List of task data
            sample_ids: List of sample IDs corresponding to each task

        Returns:
            List[int]: List of created task IDs
        """
        pass

    @abstractmethod
    def _format_task(self, samples: List[Dict]) -> Dict:
        """Format samples as an annotation task

        Args:
            samples: List of samples to include in the task

        Returns:
            Dict: Formatted task data
        """
        pass

    @abstractmethod
    def _get_task_annotation(self, task_id: int) -> Optional[Dict]:
        """Get annotation for a task if available"""
        pass

    @abstractmethod
    def _process_annotation_result(self, annotation: Dict, sample: Dict) -> Dict:
        """Process annotation result and update the sample

        Args:
            annotation: The annotation result from the annotation platform
            sample: The original sample that was annotated

        Returns:
            Dict: The updated sample with annotation results
        """
        pass

    @abstractmethod
    def _check_annotation_status(self, task_ids):
        """Check the status of annotations for the given task IDs.

        Args:
            task_ids: List of task IDs to check

        Returns:
            Tuple[bool, Dict]: (has_changes, completed_tasks_dict)
                - has_changes: If there are new annotations since last check
                - completed_tasks_dict: Dictionary for task IDs to annotations
        """
        pass

    def process_batched(self, samples):
        """Process a batch of samples by creating annotation tasks

        Args:
            samples: Dictionary of samples to process (column-oriented)

        Returns:
            Dict: Processed samples (column-oriented)
        """
        # Get dimensions of the data
        keys = list(samples.keys())
        num_samples = len(samples[keys[0]])

        # Step 1: Convert to row-oriented format (list of sample dictionaries)
        sample_list = [{key: samples[key][i] for key in keys} for i in range(num_samples)]

        # Step 2: Retrieve or generate unique IDs for each sample
        # TODO: Use ID from database or metadata if not present
        if "id" not in keys:
            sample_ids = [str(uuid.uuid4()) for _ in range(num_samples)]
        else:
            sample_ids = [sample["id"] for sample in sample_list]

        # Step 3: Process samples in batches
        for batch_start in range(0, num_samples, self.max_tasks_per_batch):
            batch_end = min(batch_start + self.max_tasks_per_batch, num_samples)

            # Prepare tasks for this batch
            tasks_data = []
            task_sample_ids = []

            for i in range(batch_start, batch_end, self.samples_per_task):
                end_idx = min(i + self.samples_per_task, batch_end)
                batch_samples = sample_list[i:end_idx]
                batch_ids = sample_ids[i:end_idx]

                # Format the samples as a task
                task_data = self._format_task(batch_samples)
                tasks_data.append(task_data)
                task_sample_ids.append(batch_ids)

            # Create and process this batch of tasks
            self._create_and_process_batch(tasks_data, task_sample_ids)

        # Step 4: Update samples with annotation results
        processed_count = 0
        for i, sample_id in enumerate(sample_ids):
            if sample_id in self.sample_to_task_id:
                task_id = self.sample_to_task_id[sample_id]

                # If waiting for annotations and they're available, add them
                if self.wait_for_annotations and task_id in self.processed_annotations:
                    annotation = self._get_task_annotation(task_id)
                    if annotation:
                        # Process the annotation result
                        sample_list[i] = self._process_annotation_result(annotation, sample_list[i])
                        processed_count += 1

        # Step 5: Convert back to column-oriented format efficiently
        # Find all keys that exist in any sample
        all_keys = set().union(*(sample.keys() for sample in sample_list))

        # Create the result dictionary with all keys
        result = {}
        for key in all_keys:
            # Use a list comprehension for better performance
            result[key] = [sample.get(key) for sample in sample_list]

        logger.info(f"Processed {num_samples} samples with {processed_count} " "annotations")
        return result

    def _create_and_process_batch(self, tasks_data, task_sample_ids):
        """Create a batch of tasks and process the results"""
        if not tasks_data:
            return

        # Generate a batch ID
        batch_id = f"batch_{self.batch_counter}_{int(time.time())}"
        self.batch_counter += 1

        # Flatten the sample IDs for this batch
        all_batch_sample_ids = [id for ids in task_sample_ids for id in ids]

        try:
            # Create the tasks in the annotation platform
            task_ids = self._create_tasks_batch(tasks_data, all_batch_sample_ids)

            # Map sample IDs to task IDs
            for task_id, sample_id_list in zip(task_ids, task_sample_ids):
                for sample_id in sample_id_list:
                    self.sample_to_task_id[sample_id] = task_id
                self.task_to_samples[task_id] = sample_id_list

                # Trigger task created event
                self.trigger_event(
                    ANNOTATION_EVENTS["TASK_CREATED"], {"task_id": task_id, "sample_ids": sample_id_list}
                )

            # Trigger batch created event
            self.trigger_event(
                ANNOTATION_EVENTS["BATCH_CREATED"],
                {"batch_id": batch_id, "task_count": len(task_ids), "sample_count": len(all_batch_sample_ids)},
            )

            # If waiting for annotations, wait for them directly
            if self.wait_for_annotations:
                try:
                    # Wait for all tasks in this batch to be annotated
                    logger.info(f"Waiting for annotations for batch {batch_id}")
                    completed_tasks = self._wait_for_batch_annotations(task_ids)
                    logger.info(
                        f"Completed {len(completed_tasks)}/{len(task_ids)} " f"annotations for batch {batch_id}"
                    )
                except TimeoutError as e:
                    # Trigger error event but continue processing
                    self.trigger_event(
                        ANNOTATION_EVENTS["ERROR_OCCURRED"],
                        {"message": (f"Timeout waiting for batch annotations: {str(e)}"), "batch_id": batch_id},
                    )

            # Trigger batch created event
            self.trigger_event(
                ANNOTATION_EVENTS["BATCH_ANNOTATION_COMPLETED"],
                {"batch_id": batch_id, "task_count": len(task_ids), "sample_count": len(all_batch_sample_ids)},
            )

        except Exception as e:
            # Trigger error event
            self.trigger_event(
                ANNOTATION_EVENTS["ERROR_OCCURRED"],
                {"message": f"Failed to create batch: {str(e)}", "batch_id": batch_id},
            )
            logger.error(f"Error creating batch {batch_id}: {str(e)}")

    def _wait_for_batch_annotations(self, task_ids=None):
        """Wait for all tasks in a batch to be annotated.

        Args:
            task_ids: List of task IDs to wait for

        Returns:
            Dict: Mapping of task IDs to their annotations
        """
        if not self.wait_for_annotations or not task_ids:
            return {}

        logger.info(f"Waiting for annotations for {len(task_ids)} tasks")

        start_time = time.time()
        completed_tasks = {}
        task_id_set = set(task_ids)
        remaining_tasks = task_id_set - set(completed_tasks.keys()) - self.processed_annotations

        # Track the last time we saw a change in annotations
        last_change_time = time.time()

        # Use efficient polling with platform-specific status checks
        while time.time() - start_time < self.timeout and remaining_tasks:
            try:
                # Check for new annotations using the platform-specific method
                has_changes, new_completed_tasks = self._check_annotation_status(list(remaining_tasks))

                # Update our completed tasks
                completed_tasks.update(new_completed_tasks)

                # Update remaining tasks and trigger events for completed tasks
                for task_id in new_completed_tasks:
                    if task_id in remaining_tasks:
                        remaining_tasks.remove(task_id)
                        self.processed_annotations.add(task_id)

                        # Trigger annotation completed event
                        annotation = new_completed_tasks[task_id]
                        self.trigger_event(
                            ANNOTATION_EVENTS["ANNOTATION_COMPLETED"],
                            {
                                "task_id": task_id,
                                "annotation_id": annotation.get("id", "unknown"),
                                "annotation": annotation,
                                "sample_ids": self.task_to_samples.get(task_id, []),
                            },
                        )

                # If no changes for a while, do a full check
                poll_timeout = max(self.poll_interval * 5, 30)
                if not has_changes and time.time() - last_change_time > poll_timeout:
                    logger.info("No new annotations detected for a while, " "performing full check")
                    last_change_time = time.time()  # Reset the timer
                elif has_changes:
                    last_change_time = time.time()

                # Log progress periodically
                logger.info(
                    f"Completed {len(completed_tasks)}/{len(task_ids)} "
                    f"annotations, {len(remaining_tasks)} remaining"
                )

                # If all tasks are annotated, we're done
                if not remaining_tasks:
                    return completed_tasks

                # Sleep before checking again
                time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Error polling for annotations: {e}")
                time.sleep(self.poll_interval)

        # If we get here, we timed out or completed all annotations
        if remaining_tasks:
            logger.warning(f"Timed out waiting for {len(remaining_tasks)} annotations")

        return completed_tasks

    def _get_project_url(self):
        """Get URL to the annotation project (abstract method)

        Returns:
            str: URL to the annotation project, or None if not available
        """
        return None

    def _get_task_url(self, task_id):
        """Get URL to a specific annotation task (abstract method)

        Args:
            task_id: ID of the task

        Returns:
            str: URL to the task, or None if not available
        """
        return None


class LabelStudioAnnotationMapper(BaseAnnotationMapper, ABC):
    """Operation for annotating data using Label Studio"""

    def __init__(self, api_url: str = None, api_key: str = None, label_config: Optional[str] = None, **kwargs):
        """Initialize the Label Studio annotation operation

        Args:
            api_url: Base URL for Label Studio API
            api_key: API key for authentication
            label_config: XML configuration for the labeling interface
            **kwargs: Additional parameters passed to BaseAnnotationMapper
        """
        # Initialize parent classes
        super().__init__(**kwargs)

        # Make sure samples_per_task is 1
        # Label studio only supports 1 sample per task
        if self.samples_per_task != 1:
            logger.warning(
                "Label Studio Annotation Mapper only supports 1 sample "
                "per task, but samples_per_task is set to "
                f"{self.samples_per_task}. Setting samples_per_task to 1."
            )
            self.samples_per_task = 1

        # Store Label Studio specific configuration
        self.api_url = api_url
        self.api_key = api_key
        self.label_config = label_config

        # Initialize Label Studio client
        try:
            self.client = label_studio_sdk.Client(url=self.api_url, api_key=self.api_key)
            logger.info(f"Connected to Label Studio at {self.api_url}")
        except ImportError:
            logger.error("Failed to import label_studio_sdk. Please install it with: " "pip install label-studio-sdk")
            raise ImportError("label-studio-sdk is required for LabelStudioAnnotationMapper")

        # Initialize project
        if self.project_id is None:
            self.project = self.setup_project()
            self.project_id = self.project.id
        else:
            try:
                self.project = self.client.get_project(self.project_id)
                logger.info(f"Using existing project with ID: {self.project_id}")
            except Exception as e:
                logger.error(f"Failed to get project with ID {self.project_id}: {e}")
                raise

    def setup_project(self):
        """Create a new project or use existing one"""
        try:
            # Create new project
            logger.info(f"Creating new Label Studio project: {self.project_name}")

            project = self.client.create_project(
                title=self.project_name, description="Created by Data Juicer", label_config=self.label_config
            )

            logger.info(f"Created new Label Studio project with ID: {project.id}")
            return project

        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            raise

    def _create_tasks_batch(self, tasks_data: List[Dict], sample_ids: List[Any]) -> List[int]:
        """Create multiple tasks in Label Studio

        Args:
            tasks_data: List of task data
            sample_ids: List of sample IDs corresponding to each task

        Returns:
            List[int]: List of created task IDs
        """
        try:
            # Create tasks in the project
            created_tasks = self.project.import_tasks(tasks_data)
            logger.debug(f"Created {len(created_tasks)} tasks in project " f"{self.project_id}")
            return created_tasks

        except Exception as e:
            logger.error(f"Failed to create tasks: {e}")
            raise

    def _check_annotation_status(self, task_ids=None):
        """Check the status of annotations for the given task IDs

        Args:
            task_ids: List of task IDs to check. If None, uses all in batch

        Returns:
            Tuple[bool, Dict]: (has_changes, completed_tasks_dict)
        """
        # Handle the case where task_ids is not provided
        if task_ids is None:
            task_ids = list(self.task_to_samples.keys())

        if not task_ids:
            return False, {}

        # Initialize tracking variables
        has_changes = False
        completed_tasks = {}

        # Filter out tasks we already know are completed
        remaining_tasks = [tid for tid in task_ids if tid not in self.processed_annotations]

        # If all tasks are already processed, return immediately
        if not remaining_tasks:
            return False, {}

        logger.debug(
            f"Checking {len(remaining_tasks)} tasks (skipping "
            f"{len(task_ids) - len(remaining_tasks)} already processed)"
        )

        try:
            # Use filters to get completed tasks efficiently
            completed_task_data = self.project.get_tasks(
                filters={
                    "conjunction": "and",
                    "items": [
                        {"filter": "filter:tasks:completed_at", "operator": "empty", "value": False, "type": "Datetime"}
                    ],
                }
            )

            # Process newly completed tasks
            for task in completed_task_data:
                task_id = task["id"]
                has_changes = True

                # Get the annotations for this task
                if "annotations" in task and task["annotations"]:
                    # Use the latest annotation
                    annotation = task["annotations"][-1]
                    completed_tasks[task_id] = annotation
                else:
                    # If task is marked as completed but has no annotations,
                    # fetch the full task to get annotations
                    full_task = self.project.get_task(task_id)
                    if "annotations" in full_task and full_task["annotations"]:
                        annotation = full_task["annotations"][-1]
                        completed_tasks[task_id] = annotation
                    else:
                        # Task is completed but has no annotations (unusual)
                        logger.warning(f"Task {task_id} is marked as completed but has " "no annotations")
                        completed_tasks[task_id] = {"id": "no_annotation", "result": []}

                logger.debug(f"Task {task_id} is newly completed")

            # Update our class-level cache of processed annotations
            for task_id in completed_tasks:
                self.processed_annotations.add(task_id)

        except Exception as e:
            logger.error(f"Error checking annotation status: {e}")

        return has_changes, completed_tasks

    def _get_task_annotation(self, task_id: int) -> Optional[Dict]:
        """Get annotation for a task if available

        Args:
            task_id: ID of the task

        Returns:
            Optional[Dict]: Annotation data or None if not yet annotated
        """
        try:
            # Get task with annotations
            task = self.project.get_task(task_id)

            # Check if task has annotations
            if task and "annotations" in task and task["annotations"]:
                return task["annotations"][0]  # Return the first annotation

            return None

        except Exception as e:
            logger.error(f"Failed to get task annotation: {e}")
            return None

    def get_all_annotations(self) -> Dict[int, Dict]:
        """Get all annotations for tasks created by this operation

        Returns:
            Dict[int, Dict]: Dictionary mapping task IDs to annotations
        """
        task_ids = list(self.task_to_samples.keys())

        annotations = {}
        for task_id in task_ids:
            annotation = self._get_task_annotation(task_id)
            if annotation:
                annotations[task_id] = annotation

        return annotations

    # for pickling
    def __getstate__(self):
        """Control how the object is pickled"""
        state = self.__dict__.copy()
        # Remove unpicklable attributes
        if "client" in state:
            del state["client"]  # Remove Label Studio client
        if "project" in state:
            del state["project"]  # Remove project reference
        return state

    # for unpickling
    def __setstate__(self, state):
        """Control how the object is unpickled"""
        self.__dict__.update(state)
        # Reconnect to Label Studio if needed
        if hasattr(self, "api_url") and hasattr(self, "api_key"):
            try:
                self.client = label_studio_sdk.Client(url=self.api_url, api_key=self.api_key)
                if hasattr(self, "project_id"):
                    self.project = self.client.get_project(self.project_id)
            except ImportError:
                pass

    def _get_project_url(self):
        """Get URL to the Label Studio project

        Returns:
            str: URL to the Label Studio project, or None if not available
        """
        if not hasattr(self, "api_url") or not self.api_url or not self.project_id:
            return None

        # Remove trailing slash if present
        base_url = self.api_url.rstrip("/")

        # Label Studio web UI URL format (remove /api if present in the URL)
        if "/api" in base_url:
            base_url = base_url.replace("/api", "")

        return f"{base_url}/projects/{self.project_id}"

    def _get_task_url(self, task_id):
        """Get URL to a specific Label Studio task

        Args:
            task_id: ID of the task

        Returns:
            str: URL to the task in Label Studio, or None if not available
        """
        if not hasattr(self, "api_url") or not self.api_url or not self.project_id:
            return None

        # Remove trailing slash if present
        base_url = self.api_url.rstrip("/")

        # Label Studio web UI URL format (remove /api if present in the URL)
        if "/api" in base_url:
            base_url = base_url.replace("/api", "")

        return f"{base_url}/projects/{self.project_id}/data?task={task_id}"
